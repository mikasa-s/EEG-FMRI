from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """4D Image to patch embedding used by NeuroSTORM."""

    def __init__(
        self,
        img_size: tuple[int, int, int, int] = (96, 96, 96, 20),
        patch_size: tuple[int, int, int, int] = (6, 6, 6, 1),
        in_chans: int = 1,
        embed_dim: int = 24,
        norm_layer=None,
        flatten: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        if len(patch_size) != 4:
            raise ValueError("patch_size must contain 4 integers: H, W, D, T")
        if patch_size[3] != 1:
            raise ValueError("NeuroSTORM does not support temporal patch merging; patch_size[3] must be 1")

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.fc = nn.Linear(
            in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3],
            out_features=embed_dim,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width, depth, timepoints = x.shape
        grid_h, grid_w, grid_d = self.grid_size
        patch_h, patch_w, patch_d, patch_t = self.patch_size

        x = x.view(batch_size, channels, grid_h, patch_h, grid_w, patch_w, grid_d, patch_d, -1, patch_t)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, patch_h * patch_w * patch_d * patch_t * channels)
        x = self.fc(x)
        x = x.view(batch_size, grid_h, grid_w, grid_d, -1, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width, depth, timepoints = x.shape
        expected = self.img_size
        if (height, width, depth, timepoints) != expected:
            raise ValueError(
                f"Input volume shape {(height, width, depth, timepoints)} does not match configured img_size {expected}"
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
