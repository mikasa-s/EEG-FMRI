from __future__ import annotations

"""对比学习骨干之上的分类头。"""

import torch
import torch.nn as nn

from .multimodal_model import EEGfMRIContrastiveModel


class EEGfMRIClassifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        finetune_cfg = cfg["finetune"]
        self.fusion = str(finetune_cfg.get("fusion", "concat"))
        # 复用对比学习阶段的双塔骨干作为下游特征提取器。
        self.backbone = EEGfMRIContrastiveModel(cfg)

        checkpoint_path = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = checkpoint.get("model", checkpoint)
            current_state = self.backbone.state_dict()
            compatible_state = {}
            # 只加载形状一致的参数，避免投影头维度变化时直接报错。
            for name, value in state.items():
                if name in current_state and current_state[name].shape == value.shape:
                    compatible_state[name] = value
            self.backbone.load_state_dict(compatible_state, strict=False)

        if bool(finetune_cfg.get("freeze_encoders", False)):
            # 冻结骨干时，只训练后接的分类头。
            for param in self.backbone.eeg_encoder.parameters():
                param.requires_grad = False
            for param in self.backbone.fmri_encoder.parameters():
                param.requires_grad = False

        eeg_dim = self.backbone.eeg_encoder.feature_dim
        fmri_dim = self.backbone.fmri_encoder.feature_dim
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
        eeg_feat = self.backbone.encode_eeg_feature(eeg) if eeg is not None else None
        fmri_feat = self.backbone.encode_fmri_feature(fmri) if fmri is not None else None

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
