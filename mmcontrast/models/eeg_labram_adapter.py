from __future__ import annotations

"""LaBraM adapter for EEG feature extraction."""

import csv
from pathlib import Path

import torch
import torch.nn as nn

from ..checkpoint_utils import load_compatible_state_dict

LABRAM_CANONICAL_CHANNEL_NAMES = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ", "OZ",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10",
    "POZ", "F1", "F2", "C1", "C2", "P1", "P2", "AF3", "AF4", "FC3",
    "FC4", "CP3", "CP4", "PO3", "PO4", "F5", "F6", "C5", "C6", "P5",
    "P6", "AF7", "AF8", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "FT9",
    "FT10", "FPZ",
]


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def _load_channel_names_from_manifest(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG channel manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names: list[str] = []
        for row in reader:
            name = str(row.get("target_channel_name", "")).strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"No target_channel_name entries found in EEG channel manifest: {path}")
    return names


class EEGLaBraMAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "labram_base_patch200_200",
        checkpoint_path: str = "",
        freeze_backbone: bool = False,
        channel_manifest_path: str = "",
    ) -> None:
        super().__init__()
        try:
            from ..backbones.eeg_labram.modeling_finetune import (
                labram_base_patch200_200,
                labram_huge_patch200_200,
                labram_large_patch200_200,
            )

            labram_factory = {
                "labram_base_patch200_200": labram_base_patch200_200,
                "labram_large_patch200_200": labram_large_patch200_200,
                "labram_huge_patch200_200": labram_huge_patch200_200,
            }
            if model_name not in labram_factory:
                raise ValueError(f"Unsupported LaBraM model_name: {model_name}")

            self.backbone = labram_factory[model_name](pretrained=False, num_classes=0)
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError("LaBraM baseline requires the 'timm' package. Please install timm>=0.9.16.") from exc
            raise

        self.feature_dim = int(getattr(self.backbone, "num_features", 200))
        self.canonical_channel_names = list(LABRAM_CANONICAL_CHANNEL_NAMES)
        self.expected_num_channels = len(self.canonical_channel_names)
        self.input_channel_names = (
            _load_channel_names_from_manifest(channel_manifest_path)
            if str(channel_manifest_path).strip()
            else []
        )

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

    def _resolve_input_chans(self, eeg: torch.Tensor) -> torch.Tensor | None:
        num_input_channels = int(eeg.shape[1])
        if self.input_channel_names:
            if len(self.input_channel_names) != num_input_channels:
                raise ValueError(
                    "LaBraM channel manifest does not match current EEG input: "
                    f"manifest has {len(self.input_channel_names)} channels but input has {num_input_channels}."
                )
            canonical_lookup = {
                _normalize_channel_name(name): index
                for index, name in enumerate(self.canonical_channel_names, start=1)
            }
            input_chans = [0]
            missing_channels: list[str] = []
            for name in self.input_channel_names:
                canonical_index = canonical_lookup.get(_normalize_channel_name(name))
                if canonical_index is None:
                    missing_channels.append(name)
                    continue
                input_chans.append(canonical_index)
            if missing_channels:
                missing_text = ", ".join(missing_channels[:8])
                raise ValueError(
                    "LaBraM channel manifest contains channels outside the supported 62-channel layout: "
                    f"{missing_text}"
                )
            return torch.tensor(input_chans, dtype=torch.long, device=eeg.device)
        if num_input_channels != self.expected_num_channels:
            raise ValueError(
                f"LaBraM baseline got {num_input_channels} EEG channels, but no channel manifest was provided "
                f"to map them into the canonical {self.expected_num_channels}-channel layout."
            )
        return None

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"LaBraM baseline expects EEG [B,C,S,P], got {tuple(eeg.shape)}")
        input_chans = self._resolve_input_chans(eeg)
        features = self.backbone.forward_features(eeg, input_chans=input_chans)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected LaBraM feature shape: {tuple(features.shape)}")
        return features
