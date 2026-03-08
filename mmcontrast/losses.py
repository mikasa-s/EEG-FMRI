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
