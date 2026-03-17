from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from ..baselines import EEGBaselineModel
from ..checkpoint_utils import extract_state_dict, filter_compatible_state_dict, load_checkpoint_file, strip_prefixes
from .eeg_cbramod_adapter import EEGCBraModAdapter
from .fmri_adapter import FMRINeuroSTORMAdapter
from .multimodal_model import EEGfMRIContrastiveModel


class EEGfMRIClassifier(nn.Module):
    @staticmethod
    def _load_submodule_checkpoint(module: nn.Module, checkpoint_path: str, prefixes: tuple[str, ...]) -> int:
        checkpoint = load_checkpoint_file(checkpoint_path)
        state_dict = extract_state_dict(checkpoint, preferred_keys=("model", "module", "state_dict"))
        best_loaded_count = 0
        for prefix in prefixes:
            subset = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
            if not subset:
                continue
            subset = strip_prefixes(subset, prefixes=("module.",))
            compatible, report = filter_compatible_state_dict(module, subset)
            module.load_state_dict(compatible, strict=False)
            best_loaded_count = max(best_loaded_count, int(report.get("loaded_count", 0)))
        return best_loaded_count

    def __init__(self, cfg: dict):
        super().__init__()
        finetune_cfg = cfg["finetune"]
        eeg_cfg = cfg["eeg_model"]
        fmri_cfg = cfg["fmri_model"]
        data_cfg = cfg["data"]

        self.fusion = str(finetune_cfg.get("fusion", "eeg_only"))
        baseline_cfg = dict(finetune_cfg.get("eeg_baseline", {}))
        self.use_eeg_baseline = bool(baseline_cfg.get("enabled", False)) and self.fusion != "fmri_only"
        self.baseline_outputs_logits = False
        self.backbone = None
        self.eeg_encoder = None
        self.fmri_encoder = None

        eeg_checkpoint_path = str(eeg_cfg.get("checkpoint_path", "")).strip()
        fmri_checkpoint_path = str(fmri_cfg.get("checkpoint_path", "")).strip()
        checkpoint_path = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()

        if self.use_eeg_baseline:
            model_name = str(baseline_cfg.get("model_name", "eegnet"))
            num_classes = int(baseline_cfg.get("num_classes", 2))
            num_channels = int(baseline_cfg.get("num_channels", 62))
            num_timepoints = int(baseline_cfg.get("num_timepoints", 200))
            if not bool(baseline_cfg.get("load_pretrained_weights", True)):
                baseline_cfg["checkpoint_path"] = ""
            baseline_init_kwargs = {
                key: value
                for key, value in baseline_cfg.items()
                if key not in {"enabled", "category", "load_pretrained_weights", "model_name", "num_classes", "num_channels", "num_timepoints"}
            }
            if str(model_name).strip().lower() == "labram":
                channel_manifest_path = str(data_cfg.get("eeg_channel_target_manifest", "")).strip()
                if not channel_manifest_path:
                    root_dir = str(data_cfg.get("root_dir", "")).strip()
                    if root_dir:
                        channel_manifest_path = str(Path(root_dir) / "eeg_channels_target.csv")
                if channel_manifest_path:
                    baseline_init_kwargs.setdefault("channel_manifest_path", channel_manifest_path)
            if str(model_name).strip().lower() not in {"labram", "cbramod"}:
                baseline_init_kwargs.pop("checkpoint_path", None)
                baseline_init_kwargs.pop("freeze_backbone", None)

            self.eeg_encoder = EEGBaselineModel(
                model_name=model_name,
                num_classes=num_classes,
                num_channels=num_channels,
                num_timepoints=num_timepoints,
                **baseline_init_kwargs,
            )
            self.baseline_outputs_logits = self.eeg_encoder.is_traditional_model()

            if self.baseline_outputs_logits and self.fusion != "eeg_only":
                raise ValueError("Traditional EEG baselines with built-in classifier only support finetune.fusion=eeg_only")
            if (not self.baseline_outputs_logits) and self.fusion != "eeg_only":
                self.fmri_encoder = FMRINeuroSTORMAdapter(**fmri_cfg)

            category = "foundation" if self.eeg_encoder.is_foundation_model() else "traditional"
            summary_parts = [f"EEG baseline: {model_name} (category={category})."]
            baseline_checkpoint_path = str(baseline_cfg.get("checkpoint_path", "")).strip()
            baseline_load_pretrained = bool(baseline_cfg.get("load_pretrained_weights", True))
            if category == "foundation":
                if baseline_load_pretrained and baseline_checkpoint_path:
                    summary_parts.append(f"Baseline pretrained checkpoint={baseline_checkpoint_path}.")
                elif baseline_load_pretrained:
                    summary_parts.append("Baseline pretrained loading enabled, but no checkpoint path was provided.")
                else:
                    summary_parts.append("Baseline pretrained loading disabled.")
            if self.fmri_encoder is not None:
                if fmri_checkpoint_path:
                    summary_parts.append(f"fMRI encoder checkpoint={fmri_checkpoint_path}.")
                else:
                    summary_parts.append("fMRI encoder initialized without checkpoint.")
            elif checkpoint_path and category == "traditional":
                summary_parts.append("Traditional EEG baseline ignores contrastive checkpoint weights.")
            self.initialization_summary = " ".join(summary_parts)
        else:
            if self.fusion == "eeg_only":
                self.eeg_encoder = EEGCBraModAdapter(**eeg_cfg)
                if checkpoint_path:
                    loaded_count = self._load_submodule_checkpoint(
                        self.eeg_encoder,
                        checkpoint_path,
                        prefixes=("eeg_encoder.", "backbone.eeg_encoder."),
                    )
                    self.initialization_summary = (
                        f"Finetune init: loaded EEG encoder from contrastive checkpoint {checkpoint_path} "
                        f"(matched_params={loaded_count})."
                    )
                elif eeg_checkpoint_path:
                    self.initialization_summary = (
                        "Finetune init: no contrastive checkpoint provided; using EEG encoder checkpoint "
                        f"({eeg_checkpoint_path})."
                    )
                else:
                    self.initialization_summary = (
                        "Finetune init: EEG-only finetune without checkpoint; using random EEG encoder initialization."
                    )
            elif self.fusion == "fmri_only":
                self.fmri_encoder = FMRINeuroSTORMAdapter(**fmri_cfg)
                if checkpoint_path:
                    loaded_count = self._load_submodule_checkpoint(
                        self.fmri_encoder,
                        checkpoint_path,
                        prefixes=("fmri_encoder.", "backbone.fmri_encoder."),
                    )
                    self.initialization_summary = (
                        f"Finetune init: loaded fMRI encoder from contrastive checkpoint {checkpoint_path} "
                        f"(matched_params={loaded_count})."
                    )
                elif fmri_checkpoint_path:
                    self.initialization_summary = (
                        "Finetune init: no contrastive checkpoint provided; using fMRI encoder checkpoint "
                        f"({fmri_checkpoint_path})."
                    )
                else:
                    self.initialization_summary = (
                        "Finetune init: fMRI-only finetune without checkpoint; using random fMRI encoder initialization."
                    )
            else:
                self.backbone = EEGfMRIContrastiveModel(cfg)
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    state = checkpoint.get("model", checkpoint)
                    current_state = self.backbone.state_dict()
                    compatible_state = {}
                    for name, value in state.items():
                        if name in current_state and current_state[name].shape == value.shape:
                            compatible_state[name] = value
                    self.backbone.load_state_dict(compatible_state, strict=False)
                    self.initialization_summary = f"Finetune init: loaded contrastive checkpoint from {checkpoint_path}."
                elif eeg_checkpoint_path or fmri_checkpoint_path:
                    sources = []
                    if eeg_checkpoint_path:
                        sources.append(f"EEG={eeg_checkpoint_path}")
                    if fmri_checkpoint_path:
                        sources.append(f"fMRI={fmri_checkpoint_path}")
                    joined_sources = ", ".join(sources)
                    self.initialization_summary = (
                        "Finetune init: no contrastive checkpoint provided; using modality-specific encoder checkpoints "
                        f"({joined_sources})."
                    )
                else:
                    self.initialization_summary = (
                        "Finetune init: no existing checkpoint provided; using random initialization and training from scratch."
                    )

        if bool(finetune_cfg.get("freeze_encoders", False)):
            if self.use_eeg_baseline:
                if self.eeg_encoder is not None and not self.baseline_outputs_logits:
                    for param in self.eeg_encoder.parameters():
                        param.requires_grad = False
                if self.fmri_encoder is not None:
                    for param in self.fmri_encoder.parameters():
                        param.requires_grad = False
            else:
                if self.eeg_encoder is not None:
                    for param in self.eeg_encoder.parameters():
                        param.requires_grad = False
                elif self.backbone is not None:
                    for param in self.backbone.eeg_encoder.parameters():
                        param.requires_grad = False
                if self.fmri_encoder is not None:
                    for param in self.fmri_encoder.parameters():
                        param.requires_grad = False
                elif self.backbone is not None:
                    for param in self.backbone.fmri_encoder.parameters():
                        param.requires_grad = False

        self.classifier = None
        if not self.baseline_outputs_logits:
            eeg_dim = 0 if self.fusion == "fmri_only" else (
                self.eeg_encoder.feature_dim if self.eeg_encoder is not None else self.backbone.eeg_encoder.feature_dim
            )
            fmri_dim = 0 if self.fusion == "eeg_only" else (
                self.fmri_encoder.feature_dim if self.fmri_encoder is not None else self.backbone.fmri_encoder.feature_dim
            )
            if self.fusion == "eeg_only":
                in_dim = eeg_dim
            elif self.fusion == "fmri_only":
                in_dim = fmri_dim
            else:
                in_dim = eeg_dim + fmri_dim

            hidden_dim = int(finetune_cfg.get("hidden_dim", 512))
            num_classes = int(finetune_cfg["num_classes"])
            dropout = float(finetune_cfg.get("dropout", 0.2))
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, eeg: torch.Tensor | None = None, fmri: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.baseline_outputs_logits:
            if eeg is None:
                raise ValueError("Traditional EEG baselines require EEG input.")
            logits = self.eeg_encoder(eeg)
            return {
                "logits": logits,
                "eeg_feat": None,
                "fmri_feat": None,
            }

        if eeg is not None:
            eeg_feat = self.eeg_encoder(eeg) if self.eeg_encoder is not None else self.backbone.encode_eeg_feature(eeg)
        else:
            eeg_feat = None
        if fmri is not None:
            fmri_feat = self.fmri_encoder(fmri) if self.fmri_encoder is not None else self.backbone.encode_fmri_feature(fmri)
        else:
            fmri_feat = None

        if self.fusion == "eeg_only":
            if eeg_feat is None:
                raise ValueError("finetune.fusion=eeg_only requires EEG input.")
            fused = eeg_feat
        elif self.fusion == "fmri_only":
            if fmri_feat is None:
                raise ValueError("finetune.fusion=fmri_only requires fMRI input.")
            fused = fmri_feat
        else:
            if eeg_feat is None or fmri_feat is None:
                raise ValueError("finetune.fusion=concat requires both EEG and fMRI input.")
            fused = torch.cat([eeg_feat, fmri_feat], dim=-1)

        logits = self.classifier(fused)
        return {
            "logits": logits,
            "eeg_feat": eeg_feat,
            "fmri_feat": fmri_feat,
        }
