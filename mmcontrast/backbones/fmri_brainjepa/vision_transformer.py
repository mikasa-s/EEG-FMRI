"""从 Brain-JEPA 项目移植并做了本地依赖清理的 Vision Transformer。"""

import importlib
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .mask_utils import apply_masks
from .tensors import repeat_interleave_batch, trunc_normal_

flash_attn_qkvpacked_func = None
try:
    # 如果环境里装了 flash_attn，就优先走更快的注意力实现；否则回退到普通实现。
    flash_attn_module = importlib.import_module("flash_attn")
    flash_attn_qkvpacked_func = getattr(flash_attn_module, "flash_attn_qkvpacked_func", None)
except ImportError:
    flash_attn_qkvpacked_func = None


class GradTs2dPE(nn.Module):
    """结合 ROI 梯度信息与二维位置编码的模块。"""

    def __init__(self, in_chan, embed_dim, grid_size, add_w=False, cls_token=False) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.grid_size = grid_size
        self.grid = self.get_grid(grid_size)
        self.emb_h = nn.Parameter(torch.zeros(grid_size[0] * grid_size[1], embed_dim // 2), requires_grad=False)
        pos_emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[0])
        self.emb_h.data.copy_(torch.from_numpy(pos_emb_h).float())

        self.add_w = add_w
        if add_w == "origin":
            self.emb_w = nn.Parameter(torch.zeros(grid_size[0] * grid_size[1], embed_dim // 2), requires_grad=False)
            pos_emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[1])
            self.emb_w.data.copy_(torch.from_numpy(pos_emb_w).float())
        if add_w == "mapping":
            self.predictor_pos_embed_proj = nn.Linear(in_chan, embed_dim // 2)
        self.cls_token = cls_token

    def get_grid(self, grid_size):
        grid_h = np.arange(grid_size[0], dtype=float)
        grid_w = np.arange(grid_size[1], dtype=float)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        return grid.reshape([2, 1, grid_size[0], grid_size[1]])

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)

    def forward(self, gradient):
        if self.add_w == "mapping":
            gradient_pos_embed = self.predictor_pos_embed_proj(gradient)
            emb_w = gradient_pos_embed.squeeze().repeat_interleave(self.grid_size[1], dim=0)
            emb_w = (emb_w - emb_w.min()) / (emb_w.max() - emb_w.min()) * 2 - 1
        elif self.add_w == "origin":
            emb_w = self.emb_w
        else:
            raise ValueError("Unsupported add_w mode")

        emb = torch.cat([self.emb_h, emb_w], dim=1).unsqueeze(0)
        if self.cls_token:
            cls = torch.zeros([1, 1, emb.shape[2]], device=emb.device, dtype=emb.dtype)
            return torch.concat([cls, emb], dim=1)
        return emb


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """标准 stochastic depth 封装。"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """Transformer block 中的前馈网络。"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """支持普通注意力和 flash attention 的多头注意力层。"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_mode="normal"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_rate = proj_drop
        self.attn_mode = attn_mode

    def forward(self, x, return_attn=None):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
        if self.attn_mode == "flash_attn" and flash_attn_qkvpacked_func is not None:
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.proj_drop_rate)
            attn = None
            if return_attn:
                x, attn, _ = flash_attn_qkvpacked_func(qkv, dropout_p=self.proj_drop_rate)
        else:
            # 默认走普通 scaled dot-product attention，兼容性最好。
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2)
        x = x.reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x, None


class Block(nn.Module):
    """标准 ViT block：注意力 + MLP。"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_mode="normal",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_mode=attn_mode,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = img_size[0] * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_2d = (img_size[0], img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        num_patches_2d,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode="normal",
        add_w=False,
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.gradient_pos_embed = gradient_pos_embed
        self.predictor_2dpe_proj = GradTs2dPE(
            gradient_pos_embed.shape[-1], predictor_embed_dim, num_patches_2d, add_w=add_w, cls_token=False
        )
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[index],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                )
                for index in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks_x, masks, return_attention=False):
        assert masks is not None and masks_x is not None, "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        batch_size = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        predictor_pos_embed = self.predictor_2dpe_proj(self.gradient_pos_embed)
        x_pos_embed = predictor_pos_embed.repeat(batch_size, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)
        _, num_context, _ = x.shape

        pos_embs = predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, batch_size, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        attn_set = []
        for block in self.predictor_blocks:
            if return_attention:
                x, attn = block(x, return_attention)
                attn_set.append(attn.detach().cpu() if attn is not None else None)
            else:
                x = block(x)
        x = self.predictor_norm(x)
        x = x[:, num_context:]
        x = self.predictor_proj(x)
        if return_attention:
            return x, attn_set
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode="normal",
        add_w=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches_2d = self.patch_embed.num_patches_2d
        self.gradient_pos_embed = gradient_pos_embed
        self.pos_embed_proj = GradTs2dPE(
            gradient_pos_embed.shape[-1], embed_dim, self.num_patches_2d, add_w=add_w, cls_token=False
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[index],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks=None, return_attention=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        x = self.patch_embed(x)
        pos_embed = self.pos_embed_proj(self.gradient_pos_embed).to(x.device, x.dtype)
        x = x + pos_embed
        if masks is not None:
            x = apply_masks(x, masks)

        attn_set = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(block, x, return_attention, use_reentrant=False)
                    attn_set.append(attn.detach().cpu() if attn is not None else None)
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = block(x, return_attention)
                    attn_set.append(attn.detach().cpu() if attn is not None else None)
                else:
                    x = block(x)
        if self.norm is not None:
            x = self.norm(x)
        if return_attention:
            return x, attn_set
        return x


def vit_predictor(**kwargs):
    return VisionTransformerPredictor(mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS = {
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
}
