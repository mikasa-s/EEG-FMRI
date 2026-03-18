from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_channel_summary import build_eeg_channel_summary
from .shared_private import EEGSharedPrivateEncoder, FMRISharedEncoder


class EEGfMRIContrastiveModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        eeg_cfg = cfg["eeg_model"]
        fmri_cfg = cfg["fmri_model"]
        train_cfg = cfg["train"]

        shared_dim = int(eeg_cfg.get("shared_dim", train_cfg.get("projection_dim", 256)))
        head_dropout = float(train_cfg.get("head_dropout", 0.0))
        self.eeg_encoder = EEGSharedPrivateEncoder(
            eeg_cfg,
            head_cfg={
                "shared_dim": shared_dim,
                "private_dim": int(eeg_cfg.get("private_dim", shared_dim)),
                "band_power_dim": int(eeg_cfg.get("band_power_dim", 5)),
                "head_dropout": head_dropout,
            },
        )
        self.fmri_encoder = FMRISharedEncoder(
            fmri_cfg,
            head_cfg={
                "shared_dim": int(fmri_cfg.get("shared_dim", shared_dim)),
                "head_dropout": head_dropout,
            },
        )

        channel_summary = build_eeg_channel_summary(cfg.get("data", {}))
        self.initialization_summary = "Contrastive model: EEG encoder=cbramod(shared/private), fMRI encoder=neurostorm(shared)."
        if channel_summary:
            self.initialization_summary = f"{self.initialization_summary} {channel_summary}"
        eeg_init_summary = str(getattr(self.eeg_encoder.backbone, "initialization_summary", "")).strip()
        fmri_init_summary = str(getattr(self.fmri_encoder.backbone, "initialization_summary", "")).strip()
        if eeg_init_summary:
            self.initialization_summary = f"{self.initialization_summary} {eeg_init_summary}"
        if fmri_init_summary:
            self.initialization_summary = f"{self.initialization_summary} {fmri_init_summary}"

    def encode_eeg_outputs(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.eeg_encoder(eeg)

    def encode_eeg_feature(self, eeg: torch.Tensor, mode: str = "concat") -> torch.Tensor:
        return self.eeg_encoder.encode_for_finetune(eeg, mode=mode)

    def encode_fmri_outputs(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.fmri_encoder(fmri)

    def encode_fmri_feature(self, fmri: torch.Tensor) -> torch.Tensor:
        return self.fmri_encoder(fmri)["fmri_shared"]

    def forward(self, eeg: torch.Tensor, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg_outputs = self.encode_eeg_outputs(eeg)
        fmri_outputs = self.encode_fmri_outputs(fmri)
        return {
            **eeg_outputs,
            **fmri_outputs,
            "eeg_embed": F.normalize(eeg_outputs["eeg_shared"], dim=-1),
            "fmri_embed": F.normalize(fmri_outputs["fmri_shared"], dim=-1),
        }
