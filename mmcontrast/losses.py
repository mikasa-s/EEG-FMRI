from __future__ import annotations

"""对比学习损失函数。"""

import torch
import torch.nn.functional as F

from .distributed import gather_with_grad, get_rank


class SymmetricInfoNCELoss(torch.nn.Module):
    """对 EEG->fMRI 和 fMRI->EEG 两个方向同时计算 InfoNCE。"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_embed: torch.Tensor, fmri_embed: torch.Tensor) -> torch.Tensor:
        """先跨卡聚合，再计算双向匹配损失。"""
        eeg_global = gather_with_grad(eeg_embed)
        fmri_global = gather_with_grad(fmri_embed)

        # 每一行都和全局 batch 做相似度，正确样本位于相同索引位置。
        logits_eeg = eeg_embed @ fmri_global.t() / self.temperature
        logits_fmri = fmri_embed @ eeg_global.t() / self.temperature

        batch = eeg_embed.size(0)
        rank = get_rank()
        labels = torch.arange(batch, device=eeg_embed.device) + rank * batch

        loss_eeg = F.cross_entropy(logits_eeg, labels)
        loss_fmri = F.cross_entropy(logits_fmri, labels)
        return 0.5 * (loss_eeg + loss_fmri)


def separation_cosine_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    shared_norm = F.normalize(shared, dim=-1)
    private_norm = F.normalize(private, dim=-1)
    cosine = (shared_norm * private_norm).sum(dim=-1)
    return (cosine ** 2).mean()


class SharedPrivatePretrainLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        band_power_weight: float = 1.0,
        separation_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.info_nce = SymmetricInfoNCELoss(temperature=temperature)
        self.band_power_weight = float(band_power_weight)
        self.separation_weight = float(separation_weight)

    def forward(
        self,
        eeg_shared: torch.Tensor,
        fmri_shared: torch.Tensor,
        eeg_private: torch.Tensor,
        band_power_pred: torch.Tensor,
        band_power_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contrastive = self.info_nce(eeg_shared, fmri_shared)
        band_power = F.mse_loss(band_power_pred, band_power_target)
        separation = separation_cosine_loss(eeg_shared, eeg_private)
        total = contrastive + (self.band_power_weight * band_power) + (self.separation_weight * separation)
        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "band_power_loss": band_power,
            "separation_loss": separation,
        }


class PureInfoNCEPretrainLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.info_nce = SymmetricInfoNCELoss(temperature=temperature)

    def forward(self, eeg_shared: torch.Tensor, fmri_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        contrastive = self.info_nce(eeg_shared, fmri_shared)
        zero = contrastive.new_zeros(())
        return {
            "loss": contrastive,
            "contrastive_loss": contrastive,
            "band_power_loss": zero,
            "separation_loss": zero,
        }


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError(f"off-diagonal expects square matrix, got {tuple(x.shape)}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsPretrainLoss(torch.nn.Module):
    def __init__(self, lambda_offdiag: float = 5e-3, eps: float = 1e-9) -> None:
        super().__init__()
        self.lambda_offdiag = float(lambda_offdiag)
        self.eps = float(eps)

    def forward(self, eeg_shared: torch.Tensor, fmri_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        if eeg_shared.ndim != 2 or fmri_shared.ndim != 2:
            raise ValueError("Barlow Twins loss expects 2D feature tensors")
        if eeg_shared.shape != fmri_shared.shape:
            raise ValueError(
                f"Barlow Twins views must have matching shapes, got {tuple(eeg_shared.shape)} and {tuple(fmri_shared.shape)}"
            )

        device_type = eeg_shared.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            z1 = eeg_shared.float()
            z2 = fmri_shared.float()
            batch_size = z1.size(0)
            if batch_size <= 1:
                loss = z1.new_zeros(())
            else:
                z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, unbiased=False, keepdim=True) + self.eps)
                z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, unbiased=False, keepdim=True) + self.eps)
                cross_correlation = (z1.T @ z2) / float(batch_size)
                on_diag = torch.diagonal(cross_correlation).add_(-1.0).pow_(2).sum()
                off_diag = _off_diagonal(cross_correlation).pow_(2).sum()
                loss = on_diag + self.lambda_offdiag * off_diag

        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "contrastive_loss": loss,
            "band_power_loss": zero,
            "separation_loss": zero,
        }
