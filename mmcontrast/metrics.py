from __future__ import annotations

"""训练与评估阶段使用的指标函数。"""

import torch


def _safe_std(values: torch.Tensor) -> float:
    """返回总体标准差，样本过少时退化为 0。"""
    values = values.float().reshape(-1)
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def contrastive_retrieval_metrics(eeg_embed: torch.Tensor, fmri_embed: torch.Tensor) -> dict[str, float]:
    """计算双向检索指标，用来衡量对齐效果。"""
    similarity = eeg_embed @ fmri_embed.t()
    targets = torch.arange(similarity.size(0), device=similarity.device)

    eeg_to_fmri_rank = similarity.argsort(dim=1, descending=True)
    fmri_to_eeg_rank = similarity.t().argsort(dim=1, descending=True)

    eeg_hits_r1 = (eeg_to_fmri_rank[:, 0] == targets).float()
    fmri_hits_r1 = (fmri_to_eeg_rank[:, 0] == targets).float()
    eeg_hits_r5 = (eeg_to_fmri_rank[:, : min(5, similarity.size(1))] == targets.unsqueeze(1)).any(dim=1).float()
    fmri_hits_r5 = (fmri_to_eeg_rank[:, : min(5, similarity.size(0))] == targets.unsqueeze(1)).any(dim=1).float()
    mean_r1_per_sample = 0.5 * (eeg_hits_r1 + fmri_hits_r1)

    eeg_r1 = float(eeg_hits_r1.mean().item())
    fmri_r1 = float(fmri_hits_r1.mean().item())
    eeg_r5 = float(eeg_hits_r5.mean().item())
    fmri_r5 = float(fmri_hits_r5.mean().item())

    return {
        "eeg_to_fmri_r1": eeg_r1,
        "eeg_to_fmri_r1_std": _safe_std(eeg_hits_r1),
        "fmri_to_eeg_r1": fmri_r1,
        "fmri_to_eeg_r1_std": _safe_std(fmri_hits_r1),
        "eeg_to_fmri_r5": eeg_r5,
        "eeg_to_fmri_r5_std": _safe_std(eeg_hits_r5),
        "fmri_to_eeg_r5": fmri_r5,
        "fmri_to_eeg_r5_std": _safe_std(fmri_hits_r5),
        "mean_r1": 0.5 * (eeg_r1 + fmri_r1),
        "mean_r1_std": _safe_std(mean_r1_per_sample),
    }


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """计算分类准确率与宏平均 F1。"""
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    accuracy_std = _safe_std((preds == labels).float())

    num_classes = int(max(labels.max().item(), preds.max().item()) + 1) if labels.numel() > 0 else 0
    f1_scores = []
    for class_id in range(num_classes):
        tp = ((preds == class_id) & (labels == class_id)).sum().item()
        fp = ((preds == class_id) & (labels != class_id)).sum().item()
        fn = ((preds != class_id) & (labels == class_id)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    macro_f1_std = 0.0 if not f1_scores else _safe_std(torch.tensor(f1_scores, dtype=torch.float32))
    return {
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "macro_f1": macro_f1,
        "macro_f1_std": macro_f1_std,
    }
