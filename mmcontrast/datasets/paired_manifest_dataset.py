from __future__ import annotations

"""基于 manifest 的 EEG-fMRI 配对数据集。"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairedEEGfMRIDataset(Dataset):
    """
    通过 CSV 清单驱动的数据集。

    默认必需列：
    - eeg_path
    - fmri_path

    可选列：
    - sample_id
    - label
    """

    def __init__(
        self,
        manifest_csv: str,
        root_dir: str = "",
        normalize_eeg: bool = True,
        normalize_fmri: bool = True,
        eeg_scale: float = 1.0,
        fmri_scale: float = 1.0,
        require_eeg: bool = True,
        require_fmri: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.normalize_eeg = normalize_eeg
        self.normalize_fmri = normalize_fmri
        self.eeg_scale = eeg_scale
        self.fmri_scale = fmri_scale
        self.require_eeg = require_eeg
        self.require_fmri = require_fmri

        self.df = pd.read_csv(manifest_csv)
        required_columns = []
        if self.require_eeg:
            required_columns.append("eeg_path")
        if self.require_fmri:
            required_columns.append("fmri_path")
        missing = [c for c in required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        """把 manifest 中的相对路径解析成真实文件路径。"""
        path = Path(p)
        if path.is_absolute() or self.root_dir is None:
            return path
        return self.root_dir / path

    def _load_array(self, path: Path) -> np.ndarray:
        """统一读取 npy、npz、pt 三种常见存储格式。"""
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".npz":
            arr = np.load(path)["arr_0"]
        elif suffix == ".pt":
            arr = torch.load(path, map_location="cpu")
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            else:
                raise ValueError(f"Unsupported .pt payload for {path}")
        else:
            raise ValueError(f"Unsupported file suffix {suffix} for {path}")
        return arr.astype(np.float32)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """对单个样本做简单 z-score 归一化。"""
        mu = x.mean()
        sigma = x.std() + 1e-6
        return (x - mu) / sigma

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """返回一条 EEG/fMRI 配对样本。"""
        row = self.df.iloc[idx]

        out = {"sample_id": str(row["sample_id"]) if "sample_id" in row else str(idx)}

        if "eeg_path" in row and not pd.isna(row["eeg_path"]):
            eeg_path = self._resolve_path(str(row["eeg_path"]))
            eeg = self._load_array(eeg_path) * self.eeg_scale

            # EEG 编码器约定输入为 [通道, 序列块, patch 长度]。
            if eeg.ndim != 3:
                raise ValueError(f"EEG sample must be 3D [C,S,P], got {eeg.shape} from {eeg_path}")
            if self.normalize_eeg:
                eeg = self._zscore(eeg)
            out["eeg"] = torch.from_numpy(eeg)
        elif self.require_eeg:
            raise ValueError("EEG path is required for this dataset but missing in manifest row.")

        if "fmri_path" in row and not pd.isna(row["fmri_path"]):
            fmri_path = self._resolve_path(str(row["fmri_path"]))
            fmri = self._load_array(fmri_path) * self.fmri_scale

            # fMRI 编码器约定输入为 [1, ROI, 时间长度]，因此二维输入会自动补通道维。
            if fmri.ndim == 2:
                fmri = np.expand_dims(fmri, axis=0)
            if fmri.ndim != 3:
                raise ValueError(f"fMRI sample must be 3D [1,ROI,T], got {fmri.shape} from {fmri_path}")
            if self.normalize_fmri:
                fmri = self._zscore(fmri)
            out["fmri"] = torch.from_numpy(fmri)
        elif self.require_fmri:
            raise ValueError("fMRI path is required for this dataset but missing in manifest row.")

        if "label" in row:
            out["label"] = int(row["label"])
        return out
