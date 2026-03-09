from __future__ import annotations

"""Internal NeuroSTORM backbone trimmed for EEG-fMRI-Contrastive."""

import itertools
from typing import Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .patchembedding import PatchEmbed

rearrange, _ = optional_import("einops", name="rearrange")

try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        """Compatibility fallback when mamba_ssm is unavailable."""

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
            super().__init__()
            inner_dim = d_model * expand
            self.in_proj = nn.Linear(d_model, inner_dim * 2)
            self.depthwise = nn.Conv1d(inner_dim, inner_dim, kernel_size=d_conv, padding=d_conv - 1, groups=inner_dim)
            self.out_proj = nn.Linear(inner_dim, d_model)
            self.activation = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate, value = self.in_proj(x).chunk(2, dim=-1)
            value = value.transpose(1, 2)
            value = self.depthwise(value)[..., : x.shape[1]].transpose(1, 2)
            value = self.activation(value)
            return self.out_proj(value * torch.sigmoid(gate))


def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    batch_size, depth, height, width, timepoints, channels = x.size()
    x = x.view(
        batch_size,
        depth // window_size[0],
        window_size[0],
        height // window_size[1],
        window_size[1],
        width // window_size[2],
        window_size[2],
        timepoints // window_size[3],
        window_size[3],
        channels,
    )
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).contiguous().view(-1, np.prod(window_size), channels)


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], dims: Sequence[int]) -> torch.Tensor:
    batch_size, depth, height, width, timepoints = dims
    x = windows.view(
        batch_size,
        torch.div(depth, window_size[0], rounding_mode="floor"),
        torch.div(height, window_size[1], rounding_mode="floor"),
        torch.div(width, window_size[2], rounding_mode="floor"),
        torch.div(timepoints, window_size[3], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    return x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(batch_size, depth, height, width, timepoints, -1)


def get_window_size(x_size: Sequence[int], window_size: Sequence[int], shift_size: Sequence[int] | None = None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for index, axis_size in enumerate(x_size):
        if axis_size <= window_size[index]:
            use_window_size[index] = axis_size
            if shift_size is not None:
                use_shift_size[index] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


class SwinTransformerBlock4D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x: torch.Tensor, mask_matrix: torch.Tensor | None) -> torch.Tensor:
        batch_size, depth, height, width, timepoints, channels = x.shape
        window_size, shift_size = get_window_size((depth, height, width, timepoints), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_depth = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_height = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_width = (window_size[2] - width % window_size[2]) % window_size[2]
        pad_time = (window_size[3] - timepoints % window_size[3]) % window_size[3]
        x = F.pad(x, (0, 0, 0, pad_time, 0, pad_width, 0, pad_height, 0, pad_depth))
        _, padded_depth, padded_height, padded_width, padded_time, _ = x.shape
        dims = [batch_size, padded_depth, padded_height, padded_width, padded_time]
        if any(shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(1, 2, 3, 4))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.mamba(x_windows)
        attn_windows = attn_windows.view(-1, *(window_size + (channels,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3]), dims=(1, 2, 3, 4))
        else:
            x = shifted_x
        if pad_depth or pad_height or pad_width or pad_time:
            x = x[:, :depth, :height, :width, :timepoints, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor | None) -> torch.Tensor:
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2) -> None:
        super().__init__()
        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, i::2, j::2, k::2, :, :] for i, j, k in itertools.product(range(2), range(2), range(2))], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


MERGING_MODE = {"mergingv2": PatchMergingV2}


def compute_mask(dims: Sequence[int], window_size: Sequence[int], shift_size: Sequence[int], device: torch.device) -> torch.Tensor:
    counter = 0
    depth, height, width, timepoints = dims
    img_mask = torch.zeros((1, depth, height, width, timepoints, 1), device=device)
    for depth_slice in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for height_slice in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for width_slice in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                for time_slice in (slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None)):
                    img_mask[:, depth_slice, height_slice, width_slice, time_slice, :] = counter
                    counter += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list[float],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(size // 2 for size in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if index % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for index in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(window_size), c_multiplier=c_multiplier) if callable(downsample) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width, timepoints = x.size()
        window_size, shift_size = get_window_size((depth, height, width, timepoints), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        padded_depth = int(np.ceil(depth / window_size[0])) * window_size[0]
        padded_height = int(np.ceil(height / window_size[1])) * window_size[1]
        padded_width = int(np.ceil(width / window_size[2])) * window_size[2]
        padded_time = int(np.ceil(timepoints / window_size[3])) * window_size[3]
        attn_mask = compute_mask([padded_depth, padded_height, padded_width, padded_time], window_size, shift_size, x.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = x.view(batch_size, depth, height, width, timepoints, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, "b d h w t c -> b c d h w t")


class BasicLayerFullAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list[float],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(size // 2 for size in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0, 0),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for index in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(window_size), c_multiplier=c_multiplier) if callable(downsample) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width, timepoints = x.size()
        window_size, _ = get_window_size((depth, height, width, timepoints), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        for block in self.blocks:
            x = block(x, None)
        x = x.view(batch_size, depth, height, width, timepoints, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, "b d h w t c -> b c d h w t")


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, patch_dim: tuple[int, int, int, int]) -> None:
        super().__init__()
        depth, height, width, timepoints = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, depth, height, width, 1))
        self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, timepoints))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed + self.time_embed[:, :, :, :, :, : x.shape[-1]]


class NeuroSTORM(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int, int, int],
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
        downsample: str | nn.Module = "mergingv2",
        **kwargs,
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=tuple(int(item) for item in patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            flatten=False,
            spatial_dims=spatial_dims,
        )
        self.grid_size = self.patch_embed.grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]

        patch_dim = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for _ in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_dim))
            pos_embed_dim *= c_multiplier
            patch_dim = (patch_dim[0] // 2, patch_dim[1] // 2, patch_dim[2] // 2, patch_dim[3])

        downsample_module = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.layers = nn.ModuleList()
        self.layers.append(
            BasicLayer(
                dim=embed_dim,
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=first_window_size,
                drop_path=dpr[: depths[0]],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=downsample_module if self.num_layers > 1 else None,
                use_checkpoint=use_checkpoint,
            )
        )
        for layer_index in range(1, self.num_layers - 1):
            self.layers.append(
                BasicLayer(
                    dim=embed_dim * (c_multiplier ** layer_index),
                    depth=depths[layer_index],
                    num_heads=num_heads[layer_index],
                    window_size=window_size,
                    drop_path=dpr[sum(depths[:layer_index]) : sum(depths[: layer_index + 1])],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    c_multiplier=c_multiplier,
                    downsample=downsample_module,
                    use_checkpoint=use_checkpoint,
                )
            )

        last_dim = embed_dim * (c_multiplier ** (self.num_layers - 1))
        last_drop_path = dpr[sum(depths[: self.num_layers - 1]) : sum(depths)]
        if last_layer_full_MSA:
            last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
                window_size[3],
            )
            final_layer = BasicLayerFullAttention(
                dim=last_dim,
                depth=depths[self.num_layers - 1],
                num_heads=num_heads[self.num_layers - 1],
                window_size=last_window_size,
                drop_path=last_drop_path,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
        else:
            final_layer = BasicLayer(
                dim=last_dim,
                depth=depths[self.num_layers - 1],
                num_heads=num_heads[self.num_layers - 1],
                window_size=window_size,
                drop_path=last_drop_path,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
        self.layers.append(final_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x.float())
        x = self.pos_drop(x)
        for layer_index in range(self.num_layers):
            x = self.pos_embeds[layer_index](x)
            x = self.layers[layer_index](x.contiguous())
        return x
