from __future__ import annotations

"""把 NeuroSTORM 封装成统一的 fMRI 编码器接口。"""

import torch
import torch.nn as nn

from ..checkpoint_utils import load_compatible_state_dict


class FMRINeuroSTORMAdapter(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "",
        img_size: tuple[int, int, int, int] = (48, 48, 48, 20),
        in_chans: int = 1,
        embed_dim: int = 24,
        window_size: tuple[int, int, int, int] = (4, 4, 4, 4),
        first_window_size: tuple[int, int, int, int] = (2, 2, 2, 2),
        patch_size: tuple[int, int, int, int] = (6, 6, 6, 1),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
        attn_drop_rate: float = 0.0,
        freeze_backbone: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.initialization_summary = "fMRI backbone init: random initialization."
        try:
            from ..backbones.fmri_neurostorm import NeuroSTORM
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", "")
            if missing_name in {"monai", "einops"}:
                raise ModuleNotFoundError(
                    "NeuroSTORM 依赖未安装。请先执行 pip install -r requirements.txt，至少安装 monai 和 einops。"
                ) from exc
            raise

        self.in_chans = int(in_chans)
        self.backbone = NeuroSTORM(
            img_size=tuple(int(item) for item in img_size),
            in_chans=self.in_chans,
            embed_dim=int(embed_dim),
            window_size=tuple(int(item) for item in window_size),
            first_window_size=tuple(int(item) for item in first_window_size),
            patch_size=tuple(int(item) for item in patch_size),
            depths=tuple(int(item) for item in depths),
            num_heads=tuple(int(item) for item in num_heads),
            c_multiplier=int(c_multiplier),
            last_layer_full_MSA=bool(last_layer_full_MSA),
            drop_rate=float(attn_drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(attn_drop_rate),
        )
        self.feature_dim = int(embed_dim) * (int(c_multiplier) ** (len(tuple(depths)) - 1))

        if checkpoint_path:
            report = load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("state_dict", "model", "backbone", "encoder"),
                prefixes=("module.", "model.", "backbone.", "encoder."),
            )
            self.initialization_summary = (
                "fMRI backbone init: loaded checkpoint "
                f"{checkpoint_path} "
                f"(loaded={int(report.get('loaded_count', 0))}, "
                f"shape_mismatch={int(report.get('skipped_shape_count', 0))}, "
                f"missing_keys={int(report.get('missing_in_checkpoint_count', 0))}, "
                f"unexpected_keys={int(report.get('skipped_missing_count', 0))})."
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        """返回对空间和时间维平均池化后的 NeuroSTORM 表征。"""
        if fmri.ndim != 6:
            raise ValueError(f"NeuroSTORM expects [B,C,H,W,D,T], got {tuple(fmri.shape)}")
        if fmri.shape[1] != self.in_chans:
            raise ValueError(f"Configured in_chans={self.in_chans}, but input has {fmri.shape[1]} channels")
        feats = self.backbone(fmri)
        return feats.mean(dim=(2, 3, 4, 5))


