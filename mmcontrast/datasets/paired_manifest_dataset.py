from __future__ import annotations

"""基于 manifest 的 EEG-fMRI 配对数据集。"""

from bisect import bisect_right
from typing import Any
import pandas as pd
from torch.utils.data import Dataset

from .sample_preparer import PairedSamplePreparer


class PairedEEGfMRIDataset(Dataset):
    """
    通过 CSV 清单驱动的数据集。

    默认必需列：
    - eeg_path
    - fmri_path

    可选列：
    - sample_id
    - label

    数据集本身只负责索引和长度，样本加载/预处理由 preparer 负责。
    subject-packed manifest 默认按样本懒加载，避免在初始化阶段把整折数据一次性拼进内存。
    """

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @classmethod
    def _manifest_is_training_ready(cls, df: pd.DataFrame) -> bool:
        if "training_ready" not in df.columns or df.empty:
            return False
        values = df["training_ready"].tolist()
        return all(cls._is_truthy(value) for value in values)

    @classmethod
    def _resolve_preload_dataset(cls, preload_dataset: bool | str, subject_packed: bool) -> bool:
        if isinstance(preload_dataset, str):
            normalized = preload_dataset.strip().lower()
            if normalized == "auto":
                return not subject_packed
            if normalized in {"1", "true", "yes", "y"}:
                return True
            if normalized in {"0", "false", "no", "n"}:
                return False
            raise ValueError(f"Unsupported preload_dataset value: {preload_dataset}")
        return bool(preload_dataset)

    @staticmethod
    def _stringify_sample_id(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

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
        preload_dataset: bool | str = "auto",
    ) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.training_ready = self._manifest_is_training_ready(self.df)
        self.preparer = PairedSamplePreparer(
            root_dir=root_dir,
            normalize_eeg=False if self.training_ready else normalize_eeg,
            normalize_fmri=False if self.training_ready else normalize_fmri,
            eeg_scale=eeg_scale,
            fmri_scale=fmri_scale,
            fmri_input_type=fmri_input_type,
            fmri_target_shape=None if self.training_ready else fmri_target_shape,
            fmri_spatial_strategy="none" if self.training_ready else fmri_spatial_strategy,
            fmri_temporal_strategy="none" if self.training_ready else fmri_temporal_strategy,
            fmri_pad_value=fmri_pad_value,
            fmri_normalize_nonzero_only=fmri_normalize_nonzero_only,
            require_eeg=require_eeg,
            require_fmri=require_fmri,
            subject_pack_cache_size=subject_pack_cache_size,
        )
        self.subject_packed = "subject_path" in self.df.columns
        self.preload_dataset = self._resolve_preload_dataset(preload_dataset, subject_packed=self.subject_packed)
        self.storage: dict[str, Any] | None = None
        self.rows = self.df.to_dict("records")
        self.subject_offsets: list[int] = []
        if self.subject_packed:
            required_columns = ["subject_path", "sample_count"]
            missing = [c for c in required_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"Subject-packed manifest missing required columns: {missing}")
        else:
            required_columns = []
            if self.preparer.require_eeg:
                required_columns.append("eeg_path")
            if self.preparer.require_fmri:
                required_columns.append("fmri_path")
            missing = [c for c in required_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"Manifest missing required columns: {missing}")

        if self.preload_dataset:
            self.storage = self.preparer.preload_dataset(self.df, subject_packed=self.subject_packed)
            self.sample_count = len(self.storage["sample_id"])
            self.is_preloaded = True
            return

        if self.subject_packed:
            running = 0
            for row in self.rows:
                running += int(row["sample_count"])
                self.subject_offsets.append(running)
            self.sample_count = running
        else:
            self.sample_count = len(self.rows)
        self.is_preloaded = False

    def __len__(self) -> int:
        return self.sample_count

    def _load_manifest_row_item(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        item: dict[str, Any] = {"sample_id": str(row.get("sample_id", idx))}
        if "subject" in row and not pd.isna(row["subject"]):
            item["subject"] = str(row["subject"])
        if "eeg_path" in row and not pd.isna(row["eeg_path"]):
            eeg_path = self.preparer.resolve_path(str(row["eeg_path"]))
            item["eeg"] = self.preparer.prepare_eeg(self.preparer.load_array(eeg_path), eeg_path)
        elif self.preparer.require_eeg:
            raise ValueError("EEG path is required for this dataset but missing in manifest row.")

        if "fmri_path" in row and not pd.isna(row["fmri_path"]):
            fmri_path = self.preparer.resolve_path(str(row["fmri_path"]))
            item["fmri"] = self.preparer.prepare_fmri(self.preparer.load_array(fmri_path), fmri_path)
        elif self.preparer.require_fmri:
            raise ValueError("fMRI path is required for this dataset but missing in manifest row.")

        if "label" in row and not pd.isna(row["label"]):
            item["label"] = int(row["label"])
        return item

    def _load_subject_packed_item(self, idx: int) -> dict[str, Any]:
        subject_index = bisect_right(self.subject_offsets, idx)
        row = self.rows[subject_index]
        previous_offset = 0 if subject_index == 0 else self.subject_offsets[subject_index - 1]
        local_idx = idx - previous_offset
        sample_count = int(row["sample_count"])
        if local_idx < 0 or local_idx >= sample_count:
            raise IndexError(f"Sample index {idx} resolved to invalid local index {local_idx} for subject row {subject_index}")

        pack_path = self.preparer.resolve_path(str(row["subject_path"]))
        pack = self.preparer.load_subject_pack(pack_path)
        item: dict[str, Any] = {
            "sample_id": self._stringify_sample_id(pack["sample_id"][local_idx]) if "sample_id" in pack else f"{subject_index}:{local_idx}"
        }
        if "subject" in row and not pd.isna(row["subject"]):
            item["subject"] = str(row["subject"])

        if self.preparer.require_eeg:
            if "eeg" not in pack:
                raise ValueError(f"Subject pack missing EEG array: {pack_path}")
            item["eeg"] = self.preparer.prepare_eeg(pack["eeg"][local_idx], pack_path)
        elif "eeg" in pack:
            item["eeg"] = self.preparer.prepare_eeg(pack["eeg"][local_idx], pack_path)

        if self.preparer.require_fmri:
            if "fmri" not in pack:
                raise ValueError(f"Subject pack missing fMRI array: {pack_path}")
            item["fmri"] = self.preparer.prepare_fmri(pack["fmri"][local_idx], pack_path)
        elif "fmri" in pack:
            item["fmri"] = self.preparer.prepare_fmri(pack["fmri"][local_idx], pack_path)

        if "labels" in pack:
            item["label"] = int(pack["labels"][local_idx])
        elif "label" in row and not pd.isna(row["label"]):
            item["label"] = int(row["label"])
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """返回一条 EEG/fMRI 配对样本。"""
        if idx < 0 or idx >= self.sample_count:
            raise IndexError(f"Sample index out of range: {idx}")

        if not self.is_preloaded:
            if self.subject_packed:
                return self._load_subject_packed_item(idx)
            return self._load_manifest_row_item(idx)

        item: dict[str, Any] = {"sample_id": self.storage["sample_id"][idx]}
        if self.storage.get("subject") is not None:
            item["subject"] = self.storage["subject"][idx]
        if self.storage.get("eeg") is not None:
            item["eeg"] = self.storage["eeg"][idx]
        if self.storage.get("fmri") is not None:
            item["fmri"] = self.storage["fmri"][idx]
        if self.storage.get("label") is not None:
            item["label"] = self.storage["label"][idx]
        return item
