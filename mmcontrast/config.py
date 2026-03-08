from __future__ import annotations

"""配置加载、落盘与基础校验逻辑。"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _parse_shape_token(shape_value: str | None) -> tuple[int, ...] | None:
    """把 manifest 里的形状字符串解析成整数元组。"""
    if shape_value is None:
        return None
    value = str(shape_value).strip().lower()
    if not value:
        return None
    tokens = [token.strip() for token in value.replace("[", "").replace("]", "").split("x") if token.strip()]
    if not tokens:
        return None
    return tuple(int(token) for token in tokens)


def _normalize_expected_shape(shape_value: Any) -> tuple[int, ...] | None:
    """兼容 list/tuple/string 三种 expected shape 写法。"""
    if shape_value is None:
        return None
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(item) for item in shape_value)
    return _parse_shape_token(str(shape_value))


def _load_array_shape(sample_path: Path) -> tuple[int, ...]:
    """在 manifest 未显式记录 shape 时，回退读取首个样本文件形状。"""
    suffix = sample_path.suffix.lower()
    if suffix == ".npy":
        return tuple(np.load(sample_path, mmap_mode="r").shape)
    if suffix == ".npz":
        with np.load(sample_path) as data:
            first_key = data.files[0]
            return tuple(data[first_key].shape)
    if suffix == ".pt":
        tensor = torch.load(sample_path, map_location="cpu")
        if hasattr(tensor, "shape"):
            return tuple(int(dim) for dim in tensor.shape)
        raise ValueError(f"PT sample does not expose shape: {sample_path}")
    raise ValueError(f"Unsupported sample format for shape validation: {sample_path}")


def _resolve_sample_shapes(manifest_path: Path, root_dir: Path | None) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """从 manifest 的首行样本解析 EEG/fMRI 形状。"""
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    eeg_shape = _parse_shape_token(first_row.get("eeg_shape"))
    fmri_shape = _parse_shape_token(first_row.get("fmri_shape"))

    if eeg_shape is None and first_row.get("eeg_path"):
        eeg_path = Path(first_row["eeg_path"])
        eeg_path = eeg_path if eeg_path.is_absolute() else (root_dir / eeg_path if root_dir else manifest_path.parent / eeg_path)
        eeg_shape = _load_array_shape(eeg_path)

    if fmri_shape is None and first_row.get("fmri_path"):
        fmri_path = Path(first_row["fmri_path"])
        fmri_path = fmri_path if fmri_path.is_absolute() else (root_dir / fmri_path if root_dir else manifest_path.parent / fmri_path)
        fmri_shape = _load_array_shape(fmri_path)

    return eeg_shape, fmri_shape


def _validate_manifest_shapes(
    manifest_path: Path,
    root_dir: Path | None,
    eeg_cfg: dict[str, Any],
    fmri_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> None:
    """确保 manifest 中的样本形状与模型配置一致。"""
    eeg_shape, fmri_shape = _resolve_sample_shapes(manifest_path, root_dir)

    expected_eeg_shape = _normalize_expected_shape(data_cfg.get("expected_eeg_shape"))
    if expected_eeg_shape is not None and eeg_shape is not None and eeg_shape != expected_eeg_shape:
        raise ValueError(
            f"EEG shape mismatch: manifest sample is {eeg_shape}, but data.expected_eeg_shape is {expected_eeg_shape}"
        )

    expected_fmri_shape = _normalize_expected_shape(data_cfg.get("expected_fmri_shape"))
    if expected_fmri_shape is not None and fmri_shape is not None and fmri_shape != expected_fmri_shape:
        raise ValueError(
            f"fMRI shape mismatch: manifest sample is {fmri_shape}, but data.expected_fmri_shape is {expected_fmri_shape}"
        )

    if eeg_shape is not None:
        if len(eeg_shape) != 3:
            raise ValueError(f"EEG samples must be [C, S, P], but got shape {eeg_shape} from {manifest_path}")
        seq_len = int(eeg_cfg.get("seq_len", eeg_shape[1]))
        in_dim = int(eeg_cfg.get("in_dim", eeg_shape[2]))
        if seq_len != eeg_shape[1]:
            raise ValueError(
                f"EEG patch count mismatch: manifest sample uses {eeg_shape[1]} patches, but eeg_model.seq_len is {seq_len}"
            )
        if in_dim != eeg_shape[2]:
            raise ValueError(
                f"EEG patch length mismatch: manifest sample patch length is {eeg_shape[2]}, but eeg_model.in_dim is {in_dim}"
            )

    if fmri_shape is not None:
        if len(fmri_shape) == 3:
            if fmri_shape[0] != 1:
                raise ValueError(f"fMRI samples with 3 dims must be [1, ROI, T], but got {fmri_shape} from {manifest_path}")
            sample_crop = (fmri_shape[1], fmri_shape[2])
        elif len(fmri_shape) == 2:
            sample_crop = (fmri_shape[0], fmri_shape[1])
        else:
            raise ValueError(f"fMRI samples must be [ROI, T] or [1, ROI, T], but got shape {fmri_shape} from {manifest_path}")

        crop_size = tuple(int(item) for item in fmri_cfg.get("crop_size", sample_crop))
        if crop_size != sample_crop:
            raise ValueError(
                f"fMRI crop size mismatch: manifest sample uses {sample_crop}, but fmri_model.crop_size is {crop_size}"
            )


@dataclass
class TrainConfig:
    """对 YAML 配置做轻量封装，避免在训练器里直接处理原始字典。"""

    raw: dict[str, Any]

    @staticmethod
    def load(path: str) -> "TrainConfig":
        """从 YAML 文件读取原始配置。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainConfig(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        """读取顶层字段。"""
        return self.raw.get(key, default)

    def section(self, key: str) -> dict[str, Any]:
        """读取某个配置分区，不存在时返回空字典。"""
        return self.raw.get(key, {})

    def dump(self, output_dir: str) -> None:
        """把解析后的配置写入输出目录，方便复现实验。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)

    def validate(self, base_dir: str | None = None) -> None:
        """检查关键路径和字段是否齐全，尽量在训练前失败而不是训练中途失败。"""
        root = Path(base_dir).resolve() if base_dir else Path.cwd()

        data_cfg = self.section("data")
        eeg_cfg = self.section("eeg_model")
        fmri_cfg = self.section("fmri_model")
        train_cfg = self.section("train")

        required_sections = ["train", "data", "eeg_model", "fmri_model"]
        for section_name in required_sections:
            if section_name not in self.raw:
                raise ValueError(f"Missing required config section: {section_name}")

        # 对比学习和微调都依赖训练集 manifest，因此这里先统一检查。
        manifest_value = str(data_cfg.get("train_manifest_csv", data_cfg.get("manifest_csv", "")))
        manifest_path = root / manifest_value
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        root_dir_value = str(data_cfg.get("root_dir", "")).strip()
        root_dir = (root / root_dir_value) if root_dir_value else None
        if root_dir is not None and not root_dir.exists():
            raise FileNotFoundError(f"Dataset root_dir not found: {root_dir}")

        for split_key in ["val_manifest_csv", "test_manifest_csv"]:
            split_path = str(data_cfg.get(split_key, "")).strip()
            if split_path:
                resolved = root / split_path
                if not resolved.exists():
                    raise FileNotFoundError(f"Manifest CSV not found: {resolved}")

        gradient_path = root / str(fmri_cfg.get("gradient_csv_path", ""))
        if not gradient_path.exists():
            raise FileNotFoundError(f"Gradient CSV not found: {gradient_path}")

        for checkpoint_key, cfg_section in [("EEG", eeg_cfg), ("fMRI", fmri_cfg)]:
            checkpoint_path = str(cfg_section.get("checkpoint_path", "")).strip()
            if checkpoint_path:
                resolved = root / checkpoint_path
                if not resolved.exists():
                    raise FileNotFoundError(f"{checkpoint_key} checkpoint not found: {resolved}")

        resume_path = str(train_cfg.get("resume_path", "")).strip()
        if resume_path:
            resolved = root / resume_path
            if not resolved.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resolved}")

        if "finetune" in self.raw:
            finetune_cfg = self.section("finetune")
            if "num_classes" not in finetune_cfg:
                raise ValueError("Missing finetune.num_classes in config")
            contrastive_checkpoint = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()
            if contrastive_checkpoint:
                resolved = root / contrastive_checkpoint
                if not resolved.exists():
                    raise FileNotFoundError(f"Contrastive checkpoint not found: {resolved}")

        _validate_manifest_shapes(manifest_path=manifest_path, root_dir=root_dir, eeg_cfg=eeg_cfg, fmri_cfg=fmri_cfg, data_cfg=data_cfg)
