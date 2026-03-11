from __future__ import annotations

"""基于 manifest 的 EEG-fMRI 配对数据集。"""

from pathlib import Path
from typing import Any
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fmri_volume_ops import ensure_volume_channel_first, resize_volume_by_strategy, zscore_volume


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
        fmri_input_type: str = "volume",
        fmri_target_shape: list[int] | tuple[int, ...] | None = None,
        fmri_spatial_strategy: str = "none",
        fmri_temporal_strategy: str = "none",
        fmri_pad_value: float = 0.0,
        fmri_normalize_nonzero_only: bool = True,
        require_eeg: bool = True,
        require_fmri: bool = True,
        subject_pack_cache_size: int = 5,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.normalize_eeg = normalize_eeg
        self.normalize_fmri = normalize_fmri
        self.eeg_scale = eeg_scale
        self.fmri_scale = fmri_scale
        self.fmri_input_type = str(fmri_input_type).strip().lower()
        self.fmri_target_shape = tuple(int(item) for item in fmri_target_shape) if fmri_target_shape else None
        self.fmri_spatial_strategy = str(fmri_spatial_strategy).strip().lower()
        self.fmri_temporal_strategy = str(fmri_temporal_strategy).strip().lower()
        self.fmri_pad_value = float(fmri_pad_value)
        self.fmri_normalize_nonzero_only = bool(fmri_normalize_nonzero_only)
        self.require_eeg = require_eeg
        self.require_fmri = require_fmri
        self.subject_pack_cache_size = max(0, int(subject_pack_cache_size))
        self._subject_pack_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        self.df = pd.read_csv(manifest_csv)
        self.subject_packed = "subject_path" in self.df.columns
        if self.subject_packed:
            required_columns = ["subject_path", "sample_count"]
            missing = [c for c in required_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"Subject-packed manifest missing required columns: {missing}")
            self.sample_index: list[tuple[int, int]] = []
            for row_idx, row in self.df.iterrows():
                sample_count = int(row["sample_count"])
                for local_idx in range(sample_count):
                    self.sample_index.append((row_idx, local_idx))
        else:
            required_columns = []
            if self.require_eeg:
                required_columns.append("eeg_path")
            if self.require_fmri:
                required_columns.append("fmri_path")
            missing = [c for c in required_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"Manifest missing required columns: {missing}")

    def __len__(self) -> int:
        if self.subject_packed:
            return len(self.sample_index)
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
            arr = np.load(path, mmap_mode="r")
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
        return arr.astype(np.float32, copy=False)

    def _load_subject_pack(self, path: Path) -> dict[str, np.ndarray]:
        suffix = path.suffix.lower()
        cache_key = str(path)
        if self.subject_pack_cache_size > 0 and cache_key in self._subject_pack_cache:
            cached = self._subject_pack_cache.pop(cache_key)
            self._subject_pack_cache[cache_key] = cached
            return cached

        if path.is_dir():
            pack = {}
            for array_path in sorted(path.glob("*.npy")):
                pack[array_path.stem] = np.load(array_path, mmap_mode="r", allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                pack = {key: data[key] for key in data.files}
        else:
            raise ValueError(f"Subject-packed files must be a directory of .npy files or a legacy .npz file, got {path}")

        if self.subject_pack_cache_size > 0:
            self._subject_pack_cache[cache_key] = pack
            while len(self._subject_pack_cache) > self.subject_pack_cache_size:
                self._subject_pack_cache.popitem(last=False)
        return pack

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """对单个样本做简单 z-score 归一化。"""
        mu = x.mean()
        sigma = x.std() + 1e-6
        return (x - mu) / sigma

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """返回一条 EEG/fMRI 配对样本。"""
        if self.subject_packed:
            row_idx, local_idx = self.sample_index[idx]
            row = self.df.iloc[row_idx]
            pack_path = self._resolve_path(str(row["subject_path"]))
            pack = self._load_subject_pack(pack_path)

            out = {
                "sample_id": str(pack["sample_id"][local_idx]) if "sample_id" in pack else f"{row_idx}:{local_idx}",
                "subject": str(row.get("subject", "")),
            }

            if self.require_eeg:
                if "eeg" not in pack:
                    raise ValueError(f"Subject pack missing EEG array: {pack_path}")
                eeg = np.asarray(pack["eeg"][local_idx], dtype=np.float32) * self.eeg_scale
                if eeg.ndim != 3:
                    raise ValueError(f"EEG sample must be 3D [C,S,P], got {eeg.shape} from {pack_path}")
                if self.normalize_eeg:
                    eeg = self._zscore(eeg)
                out["eeg"] = torch.from_numpy(eeg)

            if self.require_fmri:
                if "fmri" not in pack:
                    raise ValueError(f"Subject pack missing fMRI array: {pack_path}")
                fmri = np.asarray(pack["fmri"][local_idx], dtype=np.float32) * self.fmri_scale
                if self.fmri_input_type == "matrix":
                    if fmri.ndim == 2:
                        fmri = np.expand_dims(fmri, axis=0)
                    if fmri.ndim != 3:
                        raise ValueError(f"fMRI matrix sample must be [1,ROI,T], got {fmri.shape} from {pack_path}")
                    if self.normalize_fmri:
                        fmri = self._zscore(fmri)
                elif self.fmri_input_type == "volume":
                    fmri = ensure_volume_channel_first(fmri)
                    fmri = resize_volume_by_strategy(
                        fmri,
                        target_shape=self.fmri_target_shape,
                        spatial_strategy=self.fmri_spatial_strategy,
                        temporal_strategy=self.fmri_temporal_strategy,
                        pad_value=self.fmri_pad_value,
                    )
                    if self.normalize_fmri:
                        fmri = zscore_volume(fmri, nonzero_only=self.fmri_normalize_nonzero_only)
                else:
                    raise ValueError(f"Unsupported fmri_input_type: {self.fmri_input_type}")
                out["fmri"] = torch.from_numpy(fmri)

            if "labels" in pack:
                out["label"] = int(pack["labels"][local_idx])
            elif "label" in row:
                out["label"] = int(row["label"])
            return out

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

            if self.fmri_input_type == "matrix":
                if fmri.ndim == 2:
                    fmri = np.expand_dims(fmri, axis=0)
                if fmri.ndim != 3:
                    raise ValueError(f"fMRI matrix sample must be [1,ROI,T], got {fmri.shape} from {fmri_path}")
                if self.normalize_fmri:
                    fmri = self._zscore(fmri)
            elif self.fmri_input_type == "volume":
                fmri = ensure_volume_channel_first(fmri)
                fmri = resize_volume_by_strategy(
                    fmri,
                    target_shape=self.fmri_target_shape,
                    spatial_strategy=self.fmri_spatial_strategy,
                    temporal_strategy=self.fmri_temporal_strategy,
                    pad_value=self.fmri_pad_value,
                )
                if self.normalize_fmri:
                    fmri = zscore_volume(fmri, nonzero_only=self.fmri_normalize_nonzero_only)
            else:
                raise ValueError(f"Unsupported fmri_input_type: {self.fmri_input_type}")
            out["fmri"] = torch.from_numpy(fmri)
        elif self.require_fmri:
            raise ValueError("fMRI path is required for this dataset but missing in manifest row.")

        if "label" in row:
            out["label"] = int(row["label"])
        return out
