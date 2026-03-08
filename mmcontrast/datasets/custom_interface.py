from __future__ import annotations

"""给用户替换成自定义数据集的模板文件。"""

from typing import Any
import torch
from torch.utils.data import Dataset


class YourPairedDataset(Dataset):
    """
    把这个类替换成你自己的数据读取逻辑。

    每个样本应返回如下格式：
    {
      "eeg": Tensor[C, S, P],
      "fmri": Tensor[1, ROI, T],
      "sample_id": str,
      "label": int (optional)
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # 这里保留一个最小可运行骨架，方便你直接改造成自己的索引表。
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("Implement your own sample loading logic.")
