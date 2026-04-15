from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .fmri_volume_ops import ensure_volume_channel_first, resize_volume_by_strategy


class PairedSamplePreparer:
    """负责样本路径解析、缓存、加载与模态预处理。"""

    def __init__(
        self,
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
        require_band_power: bool = False,
        subject_pack_cache_size: int = 5,
        eeg_channel_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.normalize_eeg = bool(normalize_eeg)
        self.normalize_fmri = bool(normalize_fmri)
        self.eeg_scale = float(eeg_scale)
        self.fmri_scale = float(fmri_scale)
        self.fmri_input_type = str(fmri_input_type).strip().lower()
        self.fmri_target_shape = tuple(int(item) for item in fmri_target_shape) if fmri_target_shape else None
        self.fmri_spatial_strategy = str(fmri_spatial_strategy).strip().lower()
        self.fmri_temporal_strategy = str(fmri_temporal_strategy).strip().lower()
        self.fmri_pad_value = float(fmri_pad_value)
        self.fmri_normalize_nonzero_only = bool(fmri_normalize_nonzero_only)
        self.require_eeg = bool(require_eeg)
        self.require_fmri = bool(require_fmri)
        self.require_band_power = bool(require_band_power)
        self.subject_pack_cache_size = max(0, int(subject_pack_cache_size))
        self.eeg_channel_indices = tuple(int(index) for index in eeg_channel_indices) if eeg_channel_indices else None
        self._subject_pack_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def select_eeg_channels(self, eeg: np.ndarray, source: Path) -> np.ndarray:
        if self.eeg_channel_indices is None:
            return eeg
        if eeg.ndim < 2:
            raise ValueError(f"EEG sample must have channel axis, got {eeg.shape} from {source}")
        channel_count = int(eeg.shape[1])
        invalid = [index for index in self.eeg_channel_indices if index < 0 or index >= channel_count]
        if invalid:
            raise ValueError(f"EEG channel subset indices out of range for {source}: {invalid} with channel_count={channel_count}")
        return np.asarray(eeg[:, self.eeg_channel_indices, ...], dtype=np.float32)

    def resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute() or self.root_dir is None:
            return path
        return self.root_dir / path

    def load_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r")
        elif suffix == ".npz":
            array = np.load(path)["arr_0"]
        elif suffix == ".pt":
            array = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(array, torch.Tensor):
                array = array.numpy()
            else:
                raise ValueError(f"Unsupported .pt payload for {path}")
        else:
            raise ValueError(f"Unsupported file suffix {suffix} for {path}")
        return array.astype(np.float32, copy=False)

    def load_subject_pack(self, path: Path) -> dict[str, np.ndarray]:
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
    def zscore_sample(array: np.ndarray) -> np.ndarray:
        mean = array.mean()
        std = array.std() + 1e-6
        return (array - mean) / std

    @staticmethod
    def zscore_batch(array: np.ndarray) -> np.ndarray:
        reduce_axes = tuple(range(1, array.ndim))
        mean = array.mean(axis=reduce_axes, keepdims=True)
        std = array.std(axis=reduce_axes, keepdims=True) + 1e-6
        return (array - mean) / std

    @staticmethod
    def ensure_volume_batch_channel_first(fmri: np.ndarray) -> np.ndarray:
        if fmri.ndim == 5:
            return np.expand_dims(fmri, axis=1)
        if fmri.ndim == 6:
            return fmri
        raise ValueError(f"fMRI volume batch must be [N,H,W,D,T] or [N,C,H,W,D,T], got {fmri.shape}")

    def zscore_volume_batch(self, fmri: np.ndarray) -> np.ndarray:
        if not self.fmri_normalize_nonzero_only:
            return self.zscore_batch(fmri)

        output = np.array(fmri, copy=True)
        flat = output.reshape(output.shape[0], -1)
        mask = np.abs(flat) > 1e-8
        counts = mask.sum(axis=1)
        valid = counts > 0
        if not np.any(valid):
            return output.astype(np.float32, copy=False)

        sums = np.where(mask, flat, 0.0).sum(axis=1)
        means = np.zeros(output.shape[0], dtype=np.float32)
        means[valid] = sums[valid] / counts[valid]
        centered = np.where(mask, flat - means[:, None], 0.0)
        sq_sums = (centered * centered).sum(axis=1)
        stds = np.ones(output.shape[0], dtype=np.float32)
        stds[valid] = np.sqrt(sq_sums[valid] / counts[valid]) + 1e-6
        normalized = np.where(mask, centered / stds[:, None], 0.0)
        return normalized.reshape(output.shape).astype(np.float32, copy=False)

    def prepare_eeg_batch(self, eeg: np.ndarray, source: Path) -> torch.Tensor:
        output = np.array(eeg, dtype=np.float32, copy=True) * self.eeg_scale
        if output.ndim != 4:
            raise ValueError(f"EEG batch must be 4D [N,C,S,P], got {output.shape} from {source}")
        output = self.select_eeg_channels(output, source)
        if self.normalize_eeg:
            output = self.zscore_batch(output)
        return torch.from_numpy(output)

    def prepare_fmri_batch(self, fmri: np.ndarray, source: Path) -> torch.Tensor:
        output = np.array(fmri, dtype=np.float32, copy=True) * self.fmri_scale
        if self.fmri_input_type == "matrix":
            if output.ndim == 3:
                output = np.expand_dims(output, axis=1)
            if output.ndim != 4:
                raise ValueError(f"fMRI matrix batch must be [N,ROI,T] or [N,1,ROI,T], got {output.shape} from {source}")
            if self.normalize_fmri:
                output = self.zscore_batch(output)
            return torch.from_numpy(output)

        if self.fmri_input_type != "volume":
            raise ValueError(f"Unsupported fmri_input_type: {self.fmri_input_type}")

        output = self.ensure_volume_batch_channel_first(output)
        if self.fmri_target_shape is not None:
            if output.shape[2:] != self.fmri_target_shape:
                resized = [
                    resize_volume_by_strategy(
                        sample,
                        target_shape=self.fmri_target_shape,
                        spatial_strategy=self.fmri_spatial_strategy,
                        temporal_strategy=self.fmri_temporal_strategy,
                        pad_value=self.fmri_pad_value,
                    )
                    for sample in output
                ]
                output = np.stack(resized, axis=0)
        if self.normalize_fmri:
            output = self.zscore_volume_batch(output)
        return torch.from_numpy(output.astype(np.float32, copy=False))

    def prepare_eeg(self, eeg: np.ndarray, source: Path) -> torch.Tensor:
        output = self.prepare_eeg_batch(np.expand_dims(eeg, axis=0), source)
        return output[0]

    def prepare_fmri(self, fmri: np.ndarray, source: Path) -> torch.Tensor:
        if self.fmri_input_type == "matrix":
            output = self.prepare_fmri_batch(np.expand_dims(fmri, axis=0), source)
            return output[0]

        output = self.prepare_fmri_batch(np.expand_dims(ensure_volume_channel_first(fmri), axis=0), source)
        return output[0]

    def prepare_band_power(self, band_power: np.ndarray, source: Path) -> torch.Tensor:
        output = np.array(band_power, dtype=np.float32, copy=True)
        if output.ndim != 1:
            raise ValueError(f"Band-power target must be 1D [5], got {output.shape} from {source}")
        return torch.from_numpy(output)

    def prepare_band_power_batch(self, band_power: np.ndarray, source: Path) -> torch.Tensor:
        output = np.array(band_power, dtype=np.float32, copy=True)
        if output.ndim != 2:
            raise ValueError(f"Band-power batch must be 2D [N,5], got {output.shape} from {source}")
        return torch.from_numpy(output)

    def preload_subject_rows(self, df: pd.DataFrame) -> dict[str, Any]:
        eeg_tensors: list[torch.Tensor] = []
        fmri_tensors: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        band_power_tensors: list[torch.Tensor] = []
        sample_ids: list[str] = []
        subjects: list[str] = []

        for row_idx, row in df.iterrows():
            pack_path = self.resolve_path(str(row["subject_path"]))
            pack = self.load_subject_pack(pack_path)
            sample_count = int(row["sample_count"])
            subject_value = str(row.get("subject", ""))

            if self.require_eeg:
                if "eeg" not in pack:
                    raise ValueError(f"Subject pack missing EEG array: {pack_path}")
                eeg_tensors.append(self.prepare_eeg_batch(pack["eeg"][:sample_count], pack_path))

            if self.require_fmri:
                if "fmri" not in pack:
                    raise ValueError(f"Subject pack missing fMRI array: {pack_path}")
                fmri_tensors.append(self.prepare_fmri_batch(pack["fmri"][:sample_count], pack_path))

            if self.require_band_power:
                if "band_power" not in pack:
                    raise ValueError(f"Subject pack missing band_power array: {pack_path}")
                band_power_tensors.append(self.prepare_band_power_batch(pack["band_power"][:sample_count], pack_path))

            if "sample_id" in pack:
                sample_ids.extend(str(value) for value in pack["sample_id"][:sample_count])
            else:
                sample_ids.extend(f"{row_idx}:{local_idx}" for local_idx in range(sample_count))

            subjects.extend([subject_value] * sample_count)

            if "labels" in pack:
                label_array = np.array(pack["labels"][:sample_count], dtype=np.int64, copy=True)
                labels.append(torch.from_numpy(label_array))
            elif "label" in row:
                labels.append(torch.full((sample_count,), int(row["label"]), dtype=torch.long))

        return {
            "sample_id": sample_ids,
            "subject": subjects,
            "eeg": torch.cat(eeg_tensors, dim=0) if eeg_tensors else None,
            "fmri": torch.cat(fmri_tensors, dim=0) if fmri_tensors else None,
            "band_power": torch.cat(band_power_tensors, dim=0) if band_power_tensors else None,
            "label": torch.cat(labels, dim=0) if labels else None,
        }

    def preload_manifest_rows(self, df: pd.DataFrame) -> dict[str, Any]:
        eeg_tensors: list[torch.Tensor] = []
        fmri_tensors: list[torch.Tensor] = []
        band_power_tensors: list[torch.Tensor] = []
        labels: list[int] = []
        sample_ids: list[str] = []

        for index, row in df.iterrows():
            sample_ids.append(str(row["sample_id"]) if "sample_id" in row else str(index))

            if "eeg_path" in row and not pd.isna(row["eeg_path"]):
                eeg_path = self.resolve_path(str(row["eeg_path"]))
                eeg_tensors.append(self.prepare_eeg(self.load_array(eeg_path), eeg_path))
            elif self.require_eeg:
                raise ValueError("EEG path is required for this dataset but missing in manifest row.")

            if "fmri_path" in row and not pd.isna(row["fmri_path"]):
                fmri_path = self.resolve_path(str(row["fmri_path"]))
                fmri_tensors.append(self.prepare_fmri(self.load_array(fmri_path), fmri_path))
            elif self.require_fmri:
                raise ValueError("fMRI path is required for this dataset but missing in manifest row.")

            if "band_power_path" in row and not pd.isna(row["band_power_path"]):
                band_power_path = self.resolve_path(str(row["band_power_path"]))
                band_power_tensors.append(self.prepare_band_power(self.load_array(band_power_path), band_power_path))
            elif self.require_band_power:
                raise ValueError("Band-power path is required for this dataset but missing in manifest row.")

            if "label" in row:
                labels.append(int(row["label"]))

        return {
            "sample_id": sample_ids,
            "subject": None,
            "eeg": torch.stack(eeg_tensors, dim=0) if eeg_tensors else None,
            "fmri": torch.stack(fmri_tensors, dim=0) if fmri_tensors else None,
            "band_power": torch.stack(band_power_tensors, dim=0) if band_power_tensors else None,
            "label": torch.tensor(labels, dtype=torch.long) if labels else None,
        }

    def preload_dataset(self, df: pd.DataFrame, subject_packed: bool) -> dict[str, Any]:
        if subject_packed:
            return self.preload_subject_rows(df)
        return self.preload_manifest_rows(df)

