"""Brain-JEPA 依赖的张量工具函数。"""

import math
from logging import getLogger

import torch

logger = getLogger()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """在无梯度上下文里执行截断正态初始化。"""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """对外暴露的截断正态初始化接口。"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, batch_size, repeat):
    """按 batch 粒度重复张量块，供 mask/predictor 分支复用。"""
    num_chunks = len(x) // batch_size
    return torch.cat(
        [
            torch.cat([x[index * batch_size: (index + 1) * batch_size] for _ in range(repeat)], dim=0)
            for index in range(num_chunks)
        ],
        dim=0,
    )
