"""Brain-JEPA 的 mask 相关工具。"""

import torch


def apply_masks(x, masks):
    """根据多个 mask 从 token 序列中抽取保留位置。"""
    all_x = []
    for mask in masks:
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
