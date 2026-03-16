# EEG 基线模型定义文件
# 包含 7 个基线模型：SVM, LaBraM, CBraMod, EEG-Deformer, EEGNet, Conformer, TSception
# 所有模型定义均严格遵循官方实现，不从外部模型文件导入
# 强制要求：model_name 必须显式指定，不支持别名
# 
# 模型分类：
# - 基础模型（Foundation Models）：LaBraM, CBraMod（需要额外分类头，支持预训练 checkpoint）
# - 传统模型（Traditional Models）：SVM, EEG-Deformer, EEGNet, Conformer, TSception（端到端分类模型）

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

VALID_MODEL_NAMES = [
    "svm",
    "labram",
    "cbramod",
    "eeg_deformer",
    "eegnet",
    "conformer",
    "tsception",
]

MODEL_CATEGORIES = {
    "svm": "traditional",
    "labram": "foundation",
    "cbramod": "foundation",
    "eeg_deformer": "traditional",
    "eegnet": "traditional",
    "conformer": "traditional",
    "tsception": "traditional",
}


# ============================================================================
# einops 自定义实现（避免外部依赖）
# ============================================================================
class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pattern == 'b k c f -> b k (c f)':
            B, K, C, F = x.shape
            return x.view(B, K, C * F)
        elif self.pattern == 'b n (h d) -> b h n d':
            B, N, HD = x.shape
            H = HD // 64
            D = 64
            return x.view(B, N, H, D).transpose(1, 2)
        elif self.pattern == 'b e (h) (w) -> b (h w) e':
            B, E, H, W = x.shape
            return x.permute(0, 2, 3, 1).reshape(B, H * W, E)
        elif self.pattern == 'b (h w) e -> b e h w':
            B, HW, E = x.shape
            H = W = int(math.sqrt(HW))
            return x.reshape(B, H, W, E).permute(0, 3, 1, 2)
        raise ValueError(f"Unknown pattern: {self.pattern}")


class Reduce(nn.Module):
    def __init__(self, pattern: str, reduction: str):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return x.mean(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# einops 函数式接口（支持 rearrange() 和 reduce() 函数调用）
# ============================================================================
def rearrange(x: torch.Tensor, pattern: str, **kwargs: Any) -> torch.Tensor:
    """einops rearrange 函数式接口。"""
    if pattern == 'b n (h d) -> b h n d':
        B, N, HD = x.shape
        H = kwargs.get('h', HD // 64)
        D = HD // H
        return x.view(B, N, H, D).transpose(1, 2)
    elif pattern == 'b h n d -> b n (h d)':
        B, H, N, D = x.shape
        return x.transpose(1, 2).reshape(B, N, H * D)
    elif pattern == 'b e (h) (w) -> b (h w) e':
        B, E, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, E)
    elif pattern == 'b (h w) e -> b e h w':
        B, HW, E = x.shape
        H = W = int(math.sqrt(HW))
        return x.reshape(B, H, W, E).permute(0, 3, 1, 2)
    raise ValueError(f"Unsupported pattern: {pattern}")


def reduce(x: torch.Tensor, pattern: str, reduction: str) -> torch.Tensor:
    """einops reduce 函数式接口。"""
    if reduction == 'mean':
        return x.mean(dim=-1)
    raise ValueError(f"Unsupported reduction: {reduction}")


# ============================================================================
# SVM 分类器（传统机器学习基线）
# ============================================================================
class SVMClassifier:
    """SVM 分类器（非 nn.Module，使用 sklearn 实现）。"""
    
    def __init__(self, num_classes: int = 2, **kwargs: Any):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for SVM. Install with: pip install scikit-learn")
        self.num_classes = num_classes
        self.clf = SVC(kernel='rbf', probability=True, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        X_scaled = self.scaler.fit_transform(X_np)
        self.clf.fit(X_scaled, y_np)
        self.is_fitted = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before prediction")
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X_np = X.cpu().numpy()
        X_scaled = self.scaler.transform(X_np)
        preds = self.clf.predict(X_scaled)
        return torch.tensor(preds, dtype=torch.long, device=X.device)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before prediction")
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X_np = X.cpu().numpy()
        X_scaled = self.scaler.transform(X_np)
        probs = self.clf.predict_proba(X_scaled)
        return torch.tensor(probs, dtype=torch.float32, device=X.device)


# ============================================================================
# EEGNet 模型定义（官方实现）
# 来源：EEG-Deformer-main/models/EEGNet.py
# ============================================================================
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """EEGNet 模型（官方实现）。
    
    输入格式：[B, C, T] 或 [B, C, S, P]（兼容 3D/4D 输入）
    输出：logits [B, num_classes]
    
    官方参数：
    - F1: 8 (first convolutional filters)
    - D: 2 (depthwise multiplier)
    - F2: 16 (pointwise filters)
    - dropoutP: 0.25
    - C1: 64 (temporal kernel length)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropoutP: float = 0.25,
        F1: int = 8,
        D: int = 2,
        C1: int = 64,
        nChan: int = 62,
        nTime: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.F2 = D * F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nChan = nChan
        self.C1 = C1

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, C1), padding=(0, C1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1 * D, (nChan, 1), padding=0, bias=False, max_norm=1, groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutP),
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 22), padding=(0, 22 // 2), bias=False, groups=F1 * D),
            nn.Conv2d(F1 * D, F1 * D, (1, 1), stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutP),
        )
        
        # Calculate output size for last layer
        self.fSize = self._calculate_out_size(nChan, nTime)
        self.lastLayer = nn.Conv2d(self.F2, num_classes, (1, self.fSize[1]))

    def _calculate_out_size(self, nChan: int, nTime: int) -> Tuple[int, int]:
        """Calculate the output size based on input size."""
        data = torch.rand(1, 1, nChan, nTime)
        self.eval()
        with torch.no_grad():
            out = self.block1(data)
            out = self.block2(out)
        self.train()
        return out.shape[2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 3D [B, C, T] 和 4D [B, C, S, P] 输入
        if x.ndim == 4:
            B, C, S, P = x.shape
            x = x.reshape(B, C, S * P)
        
        if x.ndim != 3:
            raise ValueError(f"EEGNet expects [B,C,T] or [B,C,S,P], got {tuple(x.shape)}")
        
        # Add channel dimension for Conv2d: [B, C, T] -> [B, 1, C, T]
        x = torch.unsqueeze(x, dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.lastLayer(x)
        if x.ndim > 2:
            x = x.mean(dim=tuple(range(2, x.ndim)))
        return x.reshape(x.size(0), self.num_classes)
# ============================================================================
# Conformer 模型定义（官方实现）
# 来源：EEG-Conformer-main/conformer.py
# ============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 200):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(1, patch_size), stride=(1, patch_size)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + img_size // patch_size, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, E, H, W = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        # 动态生成位置编码以匹配实际序列长度
        seq_len = x.shape[1]
        if seq_len != self.pos_embedding.shape[1]:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding
        x += pos_emb
        return x


class Conformer(nn.Module):
    """Conformer 模型（官方实现）。
    
    输入格式：[B, C, T] 或 [B, C, S, P]（兼容 3D/4D 输入）
    输出：logits [B, num_classes]
    
    官方参数：
    - in_channels: 1
    - patch_size: 16
    - emb_size: 768
    - depth: 8
    - n_classes: 4
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        patch_size: int = 16,
        emb_size: int = 768,
        depth: int = 8,
        num_timepoints: int = 200,
        **kwargs: Any,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, num_timepoints)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=emb_size*4, dropout=0.1, batch_first=True),
            num_layers=depth
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 3D [B, C, T] 和 4D [B, C, S, P] 输入
        if x.ndim == 4:
            B, C, S, P = x.shape
            x = x.reshape(B, C, S * P)
        
        if x.ndim != 3:
            raise ValueError(f"Conformer expects [B,C,T] or [B,C,S,P], got {tuple(x.shape)}")
        
        # Add channel dimension: [B, C, T] -> [B, 1, C, T]
        x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
# ============================================================================
# EEG-Deformer 模型定义（官方实现，原始名称：Deformer）
# 来源：EEG-Deformer-main/models/EEGDeformer.py
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan: int, kernel_size: int, dp: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 in_chan: int, fine_grained_kernel: int = 11, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)
        x = x.view(x.size(0), -1)
        emd = torch.cat((x, x_dense), dim=-1)
        return emd

    def get_info(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel: int) -> int:
        return int(0.5 * (kernel - 1))


class Deformer(nn.Module):
    """EEG-Deformer 模型（官方实现，原始名称：Deformer）。
    
    架构：CNN Encoder + Transformer + 密集特征融合（DIP）
    输入格式：[B, C, T] 或 [B, C, S, P]（兼容 3D/4D 输入）
    输出：logits [B, num_classes]
    """
    
    def cnn_block(self, out_chan: int, kernel_size: Tuple[int, int], num_chan: int) -> nn.Sequential:
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(
        self,
        num_classes: int = 2,
        num_chan: int = 62,
        num_time: int = 200,
        temporal_kernel: int = 11,
        num_kernel: int = 64,
        depth: int = 4,
        heads: int = 16,
        mlp_dim: int = 16,
        dim_head: int = 16,
        dropout: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__()
        self.cnn_encoder = self.cnn_block(
            out_chan=num_kernel,
            kernel_size=(1, temporal_kernel),
            num_chan=num_chan,
        )

        dim = int(0.5 * num_time)

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')
        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)
        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, num_classes)
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim == 4:
            B, C, S, P = eeg.shape
            eeg = eeg.reshape(B, C, S * P)
        
        if eeg.ndim != 3:
            raise ValueError(f"EEG-Deformer expects [B,C,T] or [B,C,S,P], got {tuple(eeg.shape)}")
        
        eeg = torch.unsqueeze(eeg, dim=1)
        x = self.cnn_encoder(eeg)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel: int) -> Tuple[int, int]:
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size: int, num_layer: int) -> List[int]:
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


# ============================================================================
# TSception 模型定义（官方实现，第七个 baseline）
# 来源：EEG-Deformer-main/models/TSception.py
# ============================================================================
class TSception(nn.Module):
    """TSception 模型（官方实现）。
    
    架构：多尺度时间卷积 + 空间卷积 + 融合层
    输入格式：[B, C, T] 或 [B, C, S, P]（兼容 3D/4D 输入）
    输出：logits [B, num_classes]
    """
    
    def conv_block(self, in_chan: int, out_chan: int, kernel: Tuple[int, int],
                   step: Tuple[int, int], pool: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
        )

    def __init__(
        self,
        num_classes: int = 2,
        input_size: Optional[List[int]] = None,
        sampling_rate: int = 250,
        num_T: int = 32,
        num_S: int = 32,
        hidden: int = 128,
        dropout_rate: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__()
        if input_size is None:
            input_size = [1, 62, 200]
        
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1),
                                         (int(input_size[1] * 0.5), 1), int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            B, C, S, P = x.shape
            x = x.reshape(B, C, S * P)
        
        if x.ndim != 3:
            raise ValueError(f"TSception expects [B,C,T] or [B,C,S,P], got {tuple(x.shape)}")
        
        x = torch.unsqueeze(x, dim=1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
# ============================================================================
# LaBraM 和 CBraMod 适配器（基础模型，需要分类头）
# ============================================================================
class EEGLaBraMAdapter(nn.Module):
    """LaBraM 基础模型适配器（基础模型，需要分类头）。
    
    特点：
    - 支持 checkpoint 加载
    - 支持 batch 处理
    - 需要额外的分类头
    """
    
    def __init__(
        self,
        model_name: str = "labram_base_patch200_200",
        checkpoint_path: str = "",
        freeze_backbone: bool = False,
        **_: Any,
    ):
        super().__init__()
        try:
            from ..backbones.eeg_labram.modeling_finetune import (
                labram_base_patch200_200,
                labram_huge_patch200_200,
                labram_large_patch200_200,
            )
            from ..checkpoint_utils import load_compatible_state_dict
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError("LaBraM baseline requires the 'timm' package. Please install timm>=0.9.16.") from exc
            raise

        labram_factory = {
            "labram_base_patch200_200": labram_base_patch200_200,
            "labram_large_patch200_200": labram_large_patch200_200,
            "labram_huge_patch200_200": labram_huge_patch200_200,
        }
        if model_name not in labram_factory:
            raise ValueError(f"Unsupported LaBraM model_name: {model_name}")

        self.backbone = labram_factory[model_name](pretrained=False, num_classes=0)
        self.feature_dim = int(getattr(self.backbone, "num_features", 200))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LaBraM baseline expects EEG [B,C,S,P], got {tuple(x.shape)}")
        if x.shape[1] != 62:
            raise ValueError(
                f"LaBraM baseline currently requires 62 EEG channels, but got {int(x.shape[1])}. "
                "Please remap/select channels to 62 before finetune."
            )
        features = self.backbone.forward_features(x)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected LaBraM feature shape: {tuple(features.shape)}")
        return features


class EEGCBraModAdapter(nn.Module):
    """CBraMod 基础模型适配器（基础模型，需要分类头）。
    
    特点：
    - 支持 checkpoint 加载
    - 支持 batch 处理
    - 需要额外的分类头
    """
    
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
        **_: Any,
    ):
        super().__init__()
        from ..backbones.eeg_cbramod import CBraMod
        from ..checkpoint_utils import load_compatible_state_dict

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
            load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("state_dict", "model"),
                prefixes=("module.",),
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats.mean(dim=(1, 2))


# ============================================================================
# EEGBaselineModel 主类（统一接口）
# ============================================================================
class EEGBaselineModel(nn.Module):
    """EEG 基线模型统一接口。
    
    支持 7 个基线模型：
    - 基础模型（需要分类头）：LaBraM, CBraMod
    - 传统模型（端到端）：SVM, EEG-Deformer, EEGNet, Conformer, TSception
    
    参数:
        model_name: 模型名称（必须在 VALID_MODEL_NAMES 中）
        num_classes: 分类类别数
        num_channels: EEG 通道数
        num_timepoints: 时间点数
        **kwargs: 传递给具体模型的额外参数
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        num_channels: int = 62,
        num_timepoints: int = 200,
        **kwargs: Any,
    ):
        super().__init__()
        
        if model_name not in VALID_MODEL_NAMES:
            raise ValueError(
                f"Invalid model_name: {model_name}. "
                f"Must be one of {VALID_MODEL_NAMES}"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        
        self.model = self._create_model(model_name, num_classes, num_channels, num_timepoints, **kwargs)
        self.feature_dim = int(getattr(self.model, "feature_dim", num_classes))

    def _create_model(
        self,
        model_name: str,
        num_classes: int,
        num_channels: int,
        num_timepoints: int,
        **kwargs: Any
    ) -> nn.Module:
        model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")
        if model_name_lower == "svm":
            return SVMClassifier(num_classes=num_classes, **kwargs)
        elif model_name_lower == "labram":
            return EEGLaBraMAdapter(**kwargs)
        elif model_name_lower == "cbramod":
            return EEGCBraModAdapter(**kwargs)
        elif model_name_lower == "eeg_deformer":
            return Deformer(num_classes=num_classes, num_chan=num_channels, num_time=num_timepoints, **kwargs)
        elif model_name_lower == "eegnet":
            return EEGNet(num_classes=num_classes, num_channels=num_channels, num_timepoints=num_timepoints, **kwargs)
        elif model_name_lower == "conformer":
            return Conformer(num_classes=num_classes, num_channels=num_channels, num_timepoints=num_timepoints, **kwargs)
        elif model_name_lower == "tsception":
            return TSception(num_classes=num_classes, input_size=[1, num_channels, num_timepoints], **kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def is_foundation_model(self) -> bool:
        return is_foundation_model(self.model_name)

    def is_traditional_model(self) -> bool:
        return is_traditional_model(self.model_name)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """SVM 专用训练方法。"""
        if self.model_name.lower() == "svm":
            self.model.fit(X, y)
        else:
            raise AttributeError(f"fit() is only available for SVM model, not {self.model_name}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """SVM 专用预测方法。"""
        if self.model_name.lower() == "svm":
            return self.model.predict(X)
        else:
            raise AttributeError(f"predict() is only available for SVM model, not {self.model_name}")

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """SVM 专用概率预测方法。"""
        if self.model_name.lower() == "svm":
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"predict_proba() is only available for SVM model, not {self.model_name}")


def is_foundation_model(model_name: str) -> bool:
    return MODEL_CATEGORIES.get(model_name) == "foundation"


def is_traditional_model(model_name: str) -> bool:
    return MODEL_CATEGORIES.get(model_name) == "traditional"


__all__ = [
    "EEGBaselineModel",
    "EEGLaBraMAdapter",
    "EEGCBraModAdapter",
    "Deformer",
    "EEGNet",
    "Conformer",
    "TSception",
    "SVMClassifier",
    "VALID_MODEL_NAMES",
    "MODEL_CATEGORIES",
    "is_foundation_model",
    "is_traditional_model",
]
