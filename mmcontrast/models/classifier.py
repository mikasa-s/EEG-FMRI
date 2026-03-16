from __future__ import annotations

"""对比学习骨干之上的分类头。"""

import torch
import torch.nn as nn

from ..baselines import EEGBaselineModel
from .fmri_adapter import FMRINeuroSTORMAdapter
from .multimodal_model import EEGfMRIContrastiveModel


class EEGfMRIClassifier(nn.Module):
    def __init__(self, cfg: dict):
        """初始化分类器。
        
        Args:
            cfg: 配置字典，包含 finetune, eeg_model, fmri_model, data 等配置
        
        基线模型配置示例：
            finetune:
                eeg_baseline:
                    enabled: true
                    model_name: "eegnet"  # 7 个选项：svm, labram, cbramod, eeg_deformer, eegnet, conformer, tsception
                    num_classes: 2
                    num_channels: 62
                    num_timepoints: 200
        """
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
            # 从配置中获取基线模型参数
            model_name = str(baseline_cfg.get("model_name", "eegnet"))
            num_classes = int(baseline_cfg.get("num_classes", 2))
            num_channels = int(baseline_cfg.get("num_channels", 62))
            num_timepoints = int(baseline_cfg.get("num_timepoints", 200))
            
            # 创建基线模型
            self.eeg_encoder = EEGBaselineModel(
                model_name=model_name,
                num_classes=num_classes,
                num_channels=num_channels,
                num_timepoints=num_timepoints,
                **baseline_cfg
            )
            
            # 根据模型类别判断是否输出 logits
            # 基础模型（foundation）：需要额外分类头，不直接输出 logits
            # 传统模型（traditional）：端到端分类，直接输出 logits
            self.baseline_outputs_logits = self.eeg_encoder.is_traditional_model()
            
            if self.baseline_outputs_logits and self.fusion != "eeg_only":
                raise ValueError("Traditional EEG baselines with built-in classifier only support finetune.fusion=eeg_only")
            if (not self.baseline_outputs_logits) and self.fusion != "eeg_only":
                self.fmri_encoder = FMRINeuroSTORMAdapter(**fmri_cfg)

            # 生成初始化摘要
            category = "foundation" if self.eeg_encoder.is_foundation_model() else "traditional"
            summary_parts = [f"EEG baseline: {model_name} (category={category})."]
            if self.fmri_encoder is not None:
                if fmri_checkpoint_path:
                    summary_parts.append(f"fMRI encoder checkpoint={fmri_checkpoint_path}.")
                else:
                    summary_parts.append("fMRI encoder initialized without checkpoint.")
            elif checkpoint_path and category == "traditional":
                summary_parts.append("Traditional EEG baseline ignores contrastive checkpoint weights.")
            self.initialization_summary = " ".join(summary_parts)
        else:
            # 复用对比学习阶段的双塔骨干作为下游特征提取器。
            self.backbone = EEGfMRIContrastiveModel(cfg)
            if checkpoint_path:
                torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = checkpoint.get("model", checkpoint)
                current_state = self.backbone.state_dict()
                compatible_state = {}
                # 只加载形状一致的参数，避免投影头维度变化时直接报错。
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
            # 冻结骨干时，只训练后接的分类头。
            if self.use_eeg_baseline:
                if self.eeg_encoder is not None and not self.baseline_outputs_logits:
                    for param in self.eeg_encoder.parameters():
                        param.requires_grad = False
                if self.fmri_encoder is not None:
                    for param in self.fmri_encoder.parameters():
                        param.requires_grad = False
            else:
                for param in self.backbone.eeg_encoder.parameters():
                    param.requires_grad = False
                for param in self.backbone.fmri_encoder.parameters():
                    param.requires_grad = False

        self.classifier = None
        if not self.baseline_outputs_logits:
            eeg_dim = 0 if self.fusion == "fmri_only" else (self.eeg_encoder.feature_dim if self.eeg_encoder is not None else self.backbone.eeg_encoder.feature_dim)
            fmri_dim = 0 if self.fusion == "eeg_only" else (self.fmri_encoder.feature_dim if self.fmri_encoder is not None else self.backbone.fmri_encoder.feature_dim)
            if self.fusion == "eeg_only":
                in_dim = eeg_dim
            elif self.fusion == "fmri_only":
                in_dim = fmri_dim
            else:
                # 默认把两种模态特征直接拼接后做分类。
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
        """按配置选择需要的模态特征并输出分类 logits。"""
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
                raise ValueError("finetune.fusion=eeg_only 时必须提供 EEG 输入。")
            fused = eeg_feat
        elif self.fusion == "fmri_only":
            if fmri_feat is None:
                raise ValueError("finetune.fusion=fmri_only 时必须提供 fMRI 输入。")
            fused = fmri_feat
        else:
            if eeg_feat is None or fmri_feat is None:
                raise ValueError("finetune.fusion=concat 时必须同时提供 EEG 和 fMRI 输入。")
            fused = torch.cat([eeg_feat, fmri_feat], dim=-1)

        logits = self.classifier(fused)
        return {
            "logits": logits,
            "eeg_feat": eeg_feat,
            "fmri_feat": fmri_feat,
        }
