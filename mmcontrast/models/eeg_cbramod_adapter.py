from __future__ import annotations

"""把 CBraMod 封装成统一的 EEG 编码器接口。"""

import torch
import torch.nn as nn

from ..backbones.eeg_cbramod import CBraMod
from ..checkpoint_utils import load_compatible_state_dict


class EEGCBraModAdapter(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "",
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.initialization_summary = "EEG backbone init: random initialization."
        # 这里直接实例化本地 vendored 的 CBraMod 骨干。
        self.backbone = CBraMod(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=seq_len,
            n_layer=n_layer,
            nhead=nhead,
        )
        self.feature_dim = out_dim

        if checkpoint_path:
            # 只加载名字和形状都匹配的参数，避免不同导出格式导致失败。
            report = load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("state_dict", "model"),
                prefixes=("module.",),
            )
            self.initialization_summary = (
                "EEG backbone init: loaded checkpoint "
                f"{checkpoint_path} "
                f"(loaded={int(report.get('loaded_count', 0))}, "
                f"shape_mismatch={int(report.get('skipped_shape_count', 0))}, "
                f"missing_keys={int(report.get('missing_in_checkpoint_count', 0))}, "
                f"unexpected_keys={int(report.get('skipped_missing_count', 0))})."
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """返回经过空间和序列维平均池化后的 EEG 表征。"""
        feats = self.backbone(eeg)
        return feats.mean(dim=(1, 2))
