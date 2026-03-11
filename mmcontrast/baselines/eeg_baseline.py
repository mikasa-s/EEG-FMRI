from __future__ import annotations

"""微调阶段可选的 EEG baseline 模型。"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from ..checkpoint_utils import extract_state_dict, filter_compatible_state_dict, load_checkpoint_file, strip_prefixes
from ..models.eeg_adapter import EEGCBraModAdapter


def flatten_eeg_to_timeseries(eeg: torch.Tensor) -> torch.Tensor:
    """把 [B,C,S,P] 折叠成传统 EEG 常用的 [B,C,T]。"""
    if eeg.ndim != 4:
        raise ValueError(f"Traditional EEG baselines expect batched EEG [B,C,S,P], got {tuple(eeg.shape)}")
    batch_size, channels, seq_len, patch_len = eeg.shape
    return eeg.reshape(batch_size, channels, seq_len * patch_len)


class TraditionalEEGConvClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        conv_dim1 = max(32, hidden_dim // 2)
        conv_dim2 = max(64, hidden_dim)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim1, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_dim1),
            nn.GELU(),
            nn.Conv1d(conv_dim1, conv_dim2, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_dim2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.feature_dim = hidden_dim

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(eeg)
        return self.classifier(features)


class TraditionalEEGLSTMClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.feature_dim = hidden_dim

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        sequence = eeg.transpose(1, 2)
        outputs, _ = self.lstm(sequence)
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)


class FoundationPatchMLPEncoder(nn.Module):
    def __init__(self, patch_dim: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(feature_dim, patch_dim)
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"Foundation EEG baselines expect batched EEG [B,C,S,P], got {tuple(eeg.shape)}")
        patch_features = self.patch_proj(eeg)
        return patch_features.mean(dim=(1, 2))


def load_contrastive_eeg_encoder_weights(model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """从对比学习 checkpoint 中提取 eeg_encoder 权重。"""
    raw_state = extract_state_dict(load_checkpoint_file(checkpoint_path), preferred_keys=("state_dict", "model"))
    normalized_state = strip_prefixes(raw_state, prefixes=("module.",))
    encoder_state = OrderedDict()
    for key, value in normalized_state.items():
        if key.startswith("eeg_encoder."):
            encoder_state[key[len("eeg_encoder."):]] = value
    compatible_state, report = filter_compatible_state_dict(model, encoder_state)
    model.load_state_dict(compatible_state, strict=False)
    return report


class EEGBaselineModel(nn.Module):
    def __init__(self, data_cfg: dict[str, Any], eeg_cfg: dict[str, Any], finetune_cfg: dict[str, Any]) -> None:
        super().__init__()
        baseline_cfg = dict(finetune_cfg.get("eeg_baseline", {}))
        self.category = str(baseline_cfg.get("category", "foundation")).strip().lower()
        self.model_name = str(baseline_cfg.get("model_name", "cbramod")).strip().lower()
        self.produces_logits = self.category == "traditional"

        expected_shape = data_cfg.get("expected_eeg_shape", [])
        if not isinstance(expected_shape, (list, tuple)) or len(expected_shape) < 3:
            raise ValueError("data.expected_eeg_shape must be configured as [C,S,P] when using finetune.eeg_baseline")

        in_channels = int(expected_shape[0])
        patch_dim = int(expected_shape[2])
        hidden_dim = int(baseline_cfg.get("feature_dim", eeg_cfg.get("out_dim", 256)))
        num_classes = int(finetune_cfg.get("num_classes", 2))
        dropout = float(baseline_cfg.get("dropout", finetune_cfg.get("dropout", 0.2)))
        self.initialization_summary = ""
        self.input_layout = "patch"
        self.feature_dim = hidden_dim

        if self.category == "traditional":
            self.input_layout = "timeseries"
            if self.model_name in {"conv1d", "cnn", "shallowconv1d"}:
                self.model = TraditionalEEGConvClassifier(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                )
            elif self.model_name in {"lstm", "bilstm"}:
                num_layers = int(baseline_cfg.get("traditional_num_layers", 2))
                self.model = TraditionalEEGLSTMClassifier(
                    in_channels=in_channels,
                    hidden_dim=int(baseline_cfg.get("traditional_hidden_dim", hidden_dim)),
                    num_layers=num_layers,
                    num_classes=num_classes,
                    dropout=dropout,
                )
            else:
                raise ValueError("Unsupported traditional EEG baseline model. Expected one of: conv1d, cnn, shallowconv1d, lstm, bilstm")
            self.initialization_summary = f"EEG baseline: category=traditional, model={self.model_name}, direct_logits=true."
        elif self.category == "foundation":
            if self.model_name == "cbramod":
                self.model = EEGCBraModAdapter(**eeg_cfg)
                self.feature_dim = int(self.model.feature_dim)
                contrastive_checkpoint = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()
                if contrastive_checkpoint:
                    report = load_contrastive_eeg_encoder_weights(self.model, contrastive_checkpoint)
                    self.initialization_summary = (
                        f"EEG baseline: category=foundation, model=cbramod, loaded contrastive EEG encoder "
                        f"({report['loaded_count']} tensors)."
                    )
                else:
                    self.initialization_summary = "EEG baseline: category=foundation, model=cbramod."
            elif self.model_name in {"patch_mlp", "mlp"}:
                self.model = FoundationPatchMLPEncoder(patch_dim=patch_dim, feature_dim=hidden_dim, dropout=dropout)
                self.feature_dim = int(self.model.feature_dim)
                self.initialization_summary = "EEG baseline: category=foundation, model=patch_mlp."
            else:
                raise ValueError("Unsupported foundation EEG baseline model. Expected one of: cbramod, patch_mlp, mlp")
        else:
            raise ValueError("finetune.eeg_baseline.category must be either 'traditional' or 'foundation'")

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "timeseries":
            eeg = flatten_eeg_to_timeseries(eeg)
        return self.model(eeg)