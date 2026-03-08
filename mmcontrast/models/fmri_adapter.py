from __future__ import annotations

"""把 Brain-JEPA ViT 封装成统一的 fMRI 编码器接口。"""

import numpy as np
import torch
import torch.nn as nn

from ..backbones.fmri_brainjepa import VIT_EMBED_DIMS
from ..backbones.fmri_brainjepa import vit_base, vit_large, vit_small
from ..checkpoint_utils import load_compatible_state_dict


MODEL_REGISTRY = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
}


class FMRIBrainJEPAAdapter(nn.Module):
    def __init__(
        self,
        gradient_csv_path: str,
        checkpoint_path: str = "",
        model_name: str = "vit_base",
        crop_size: tuple[int, int] = (450, 160),
        patch_size: int = 16,
        attn_mode: str = "normal",
        add_w: str = "mapping",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        embed_dim = VIT_EMBED_DIMS[model_name]

        # 梯度文件用于构造 Brain-JEPA 使用的位置编码映射。
        gradient = np.loadtxt(gradient_csv_path, delimiter=",", dtype=np.float32)
        gradient = torch.from_numpy(gradient).unsqueeze(0)

        model_ctor = MODEL_REGISTRY[model_name]
        # 这里把原始 Brain-JEPA 骨干包装成单一前向入口。
        self.backbone = model_ctor(
            img_size=crop_size,
            patch_size=patch_size,
            in_chans=1,
            gradient_pos_embed=gradient,
            attn_mode=attn_mode,
            add_w=add_w,
        )
        self.feature_dim = embed_dim

        if checkpoint_path:
            # Brain-JEPA checkpoint 里常见的 target_encoder/encoder 都做兼容处理。
            load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("target_encoder", "encoder", "state_dict", "model"),
                prefixes=("module.",),
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        """返回对 token 维平均池化后的 fMRI 表征。"""
        feats = self.backbone(fmri, masks=None, return_attention=False)
        return feats.mean(dim=1)
