from __future__ import annotations

"""LaBraM adapter for EEG feature extraction."""

import csv
from pathlib import Path

import torch
import torch.nn as nn

from ..checkpoint_utils import load_compatible_state_dict

JOINT_CHANNEL_MANIFEST = Path(__file__).resolve().parents[2] / "cache" / "joint_contrastive" / "eeg_channels_target.csv"

def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def _load_channel_names_from_manifest(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG channel manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names: list[str] = []
        for row in reader:
            name = str(row.get("target_channel_name", "")).strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"No target_channel_name entries found in EEG channel manifest: {path}")
    return names


def _count_common_channel_matches(current_names: list[str], common_names: list[str]) -> int:
    current_lookup = {_normalize_channel_name(name) for name in current_names}
    return sum(1 for name in common_names if _normalize_channel_name(name) in current_lookup)


class EEGLaBraMAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "labram_base_patch200_200",
        checkpoint_path: str = "",
        freeze_backbone: bool = False,
        channel_manifest_path: str = "",
    ) -> None:
        super().__init__()
        try:
            from ..backbones.eeg_labram.modeling_finetune import (
                labram_base_patch200_200,
                labram_huge_patch200_200,
                labram_large_patch200_200,
            )

            labram_factory = {
                "labram_base_patch200_200": labram_base_patch200_200,
                "labram_large_patch200_200": labram_large_patch200_200,
                "labram_huge_patch200_200": labram_huge_patch200_200,
            }
            if model_name not in labram_factory:
                raise ValueError(f"Unsupported LaBraM model_name: {model_name}")

            self.backbone = labram_factory[model_name](pretrained=False, num_classes=0)
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError("LaBraM baseline requires the 'timm' package. Please install timm>=0.9.16.") from exc
            raise

        self.feature_dim = int(getattr(self.backbone, "num_features", 200))
        self.dropped_channel_names: list[str] = []
        self.input_channel_names = (
            _load_channel_names_from_manifest(channel_manifest_path)
            if str(channel_manifest_path).strip()
            else []
        )
        self.common_channel_match_count: int | None = None
        self.common_channel_total_count: int | None = None
        if self.input_channel_names and JOINT_CHANNEL_MANIFEST.exists():
            common_channel_names = _load_channel_names_from_manifest(JOINT_CHANNEL_MANIFEST)
            self.common_channel_match_count = _count_common_channel_matches(self.input_channel_names, common_channel_names)
            self.common_channel_total_count = len(common_channel_names)

        if checkpoint_path:
            load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("model", "module", "state_dict"),
                prefixes=("module.", "model."),
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _resolve_input_chans(self, eeg: torch.Tensor) -> torch.Tensor:
        num_input_channels = int(eeg.shape[1])
        if self.input_channel_names and len(self.input_channel_names) != num_input_channels:
            raise ValueError(
                "LaBraM channel manifest does not match current EEG input: "
                f"manifest has {len(self.input_channel_names)} channels but input has {num_input_channels}."
            )
        return torch.arange(num_input_channels + 1, dtype=torch.long, device=eeg.device)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"LaBraM baseline expects EEG [B,C,S,P], got {tuple(eeg.shape)}")
        input_chans = self._resolve_input_chans(eeg)
        features = self.backbone.forward_features(eeg, input_chans=input_chans)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected LaBraM feature shape: {tuple(features.shape)}")
        return features
