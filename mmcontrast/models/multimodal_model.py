from __future__ import annotations

"""EEG 与 fMRI 双塔对比学习模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_adapter import EEGCBraModAdapter
from .fmri_adapter import FMRINeuroSTORMAdapter


class EEGfMRIContrastiveModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        eeg_cfg = cfg["eeg_model"]
        fmri_cfg = cfg["fmri_model"]
        train_cfg = cfg["train"]

        # 两个编码器各自负责把原始模态映射到特征空间。
        self.eeg_encoder = EEGCBraModAdapter(**eeg_cfg)
        self.fmri_encoder = FMRINeuroSTORMAdapter(**fmri_cfg)

        # 投影头把两种模态的特征拉到同一个对比学习空间。
        proj_dim = int(train_cfg.get("projection_dim", 256))
        self.eeg_proj = nn.Linear(self.eeg_encoder.feature_dim, proj_dim)
        self.fmri_proj = nn.Linear(self.fmri_encoder.feature_dim, proj_dim)

    def encode_eeg_feature(self, eeg: torch.Tensor) -> torch.Tensor:
        """提取 EEG 模态特征，不做投影归一化。"""
        return self.eeg_encoder(eeg)

    def encode_fmri_feature(self, fmri: torch.Tensor) -> torch.Tensor:
        """提取 fMRI 模态特征，不做投影归一化。"""
        return self.fmri_encoder(fmri)

    def project_eeg(self, eeg_feat: torch.Tensor) -> torch.Tensor:
        """把 EEG 特征映射到对比学习嵌入空间。"""
        return F.normalize(self.eeg_proj(eeg_feat), dim=-1)

    def project_fmri(self, fmri_feat: torch.Tensor) -> torch.Tensor:
        """把 fMRI 特征映射到对比学习嵌入空间。"""
        return F.normalize(self.fmri_proj(fmri_feat), dim=-1)

    def forward(self, eeg: torch.Tensor, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        """同时返回原始特征和归一化后的对比嵌入。"""
        eeg_feat = self.encode_eeg_feature(eeg)
        fmri_feat = self.encode_fmri_feature(fmri)

        eeg_embed = self.project_eeg(eeg_feat)
        fmri_embed = self.project_fmri(fmri_feat)
        return {
            "eeg_feat": eeg_feat,
            "fmri_feat": fmri_feat,
            "eeg_embed": eeg_embed,
            "fmri_embed": fmri_embed,
        }
