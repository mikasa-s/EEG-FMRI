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


def _load_subject_pack_array_shape(subject_path: Path, array_name: str) -> tuple[int, ...] | None:
    if subject_path.is_dir():
        array_path = subject_path / f"{array_name}.npy"
        if array_path.exists():
            return tuple(np.load(array_path, mmap_mode="r", allow_pickle=False).shape)
        return None

    if subject_path.suffix.lower() == ".npz":
        with np.load(subject_path, allow_pickle=False) as data:
            if array_name in data:
                return tuple(data[array_name].shape)
        return None

    raise ValueError(f"Unsupported subject-packed format for shape validation: {subject_path}")


def _resolve_sample_shapes(manifest_path: Path, root_dir: Path | None) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """从 manifest 的首行样本解析 EEG/fMRI 形状。"""
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    eeg_shape = _parse_shape_token(first_row.get("eeg_shape"))
    fmri_shape = _parse_shape_token(first_row.get("fmri_shape"))

    if (eeg_shape is None or fmri_shape is None) and first_row.get("subject_path"):
        subject_path = Path(first_row["subject_path"])
        subject_path = subject_path if subject_path.is_absolute() else (root_dir / subject_path if root_dir else manifest_path.parent / subject_path)
        if eeg_shape is None:
            eeg_pack_shape = _load_subject_pack_array_shape(subject_path, "eeg")
            if eeg_pack_shape is not None:
                eeg_shape = tuple(int(dim) for dim in eeg_pack_shape[1:]) if len(eeg_pack_shape) >= 2 else tuple(int(dim) for dim in eeg_pack_shape)
        if fmri_shape is None:
            fmri_pack_shape = _load_subject_pack_array_shape(subject_path, "fmri")
            if fmri_pack_shape is not None:
                fmri_shape = tuple(int(dim) for dim in fmri_pack_shape[1:]) if len(fmri_pack_shape) >= 2 else tuple(int(dim) for dim in fmri_pack_shape)

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

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    subject_packed = bool(first_row and first_row.get("subject_path"))

    if subject_packed and eeg_shape is not None and len(eeg_shape) >= 4:
        eeg_shape = eeg_shape[1:]
    if subject_packed and fmri_shape is not None and len(fmri_shape) >= 5:
        fmri_shape = fmri_shape[1:]

    expected_eeg_shape = _normalize_expected_shape(data_cfg.get("expected_eeg_shape"))
    if expected_eeg_shape is not None and eeg_shape is not None:
        if len(expected_eeg_shape) != len(eeg_shape):
            raise ValueError(
                f"EEG shape rank mismatch: manifest sample is {eeg_shape}, but data.expected_eeg_shape is {expected_eeg_shape}"
            )
        if len(eeg_shape) >= 2 and tuple(eeg_shape[1:]) != tuple(expected_eeg_shape[1:]):
            raise ValueError(
                f"EEG patch shape mismatch: manifest sample trailing dims are {eeg_shape[1:]}, but data.expected_eeg_shape expects {expected_eeg_shape[1:]}"
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
        fmri_input_type = str(data_cfg.get("fmri_input_type", "volume")).strip().lower()
        if fmri_input_type == "matrix":
            if len(fmri_shape) == 3:
                if fmri_shape[0] != 1:
                    raise ValueError(f"fMRI matrix samples with 3 dims must be [1, ROI, T], but got {fmri_shape} from {manifest_path}")
                sample_crop = (fmri_shape[1], fmri_shape[2])
            elif len(fmri_shape) == 2:
                sample_crop = (fmri_shape[0], fmri_shape[1])
            else:
                raise ValueError(f"fMRI matrix samples must be [ROI, T] or [1, ROI, T], but got shape {fmri_shape} from {manifest_path}")

            backbone = str(fmri_cfg.get("backbone", "neurostorm")).strip().lower()
            if backbone == "neurostorm":
                raise ValueError("data.fmri_input_type=matrix is incompatible with the active NeuroSTORM fMRI backbone")
        elif fmri_input_type == "volume":
            if len(fmri_shape) == 5:
                sample_volume = fmri_shape[1:]
                input_channels = fmri_shape[0]
            elif len(fmri_shape) == 4:
                sample_volume = fmri_shape
                input_channels = 1
            else:
                raise ValueError(f"fMRI volume samples must be [H,W,D,T] or [C,H,W,D,T], but got shape {fmri_shape} from {manifest_path}")

            configured_channels = int(fmri_cfg.get("in_chans", input_channels))
            if configured_channels != input_channels:
                raise ValueError(
                    f"fMRI channel mismatch: manifest sample uses {input_channels} channel(s), but fmri_model.in_chans is {configured_channels}"
                )

            img_size = tuple(int(item) for item in fmri_cfg.get("img_size", data_cfg.get("fmri_target_shape", sample_volume)))
            if len(img_size) != 4:
                raise ValueError(f"fmri_model.img_size must be a 4-element sequence, got {img_size}")

            target_shape = tuple(int(item) for item in data_cfg.get("fmri_target_shape", img_size))
            if len(target_shape) != 4:
                raise ValueError(f"data.fmri_target_shape must be a 4-element sequence, got {target_shape}")

            if target_shape != img_size:
                raise ValueError(f"data.fmri_target_shape {target_shape} must match fmri_model.img_size {img_size}")

            spatial_strategy = str(data_cfg.get("fmri_spatial_strategy", "none")).strip().lower()
            temporal_strategy = str(data_cfg.get("fmri_temporal_strategy", "none")).strip().lower()
            if spatial_strategy == "none" and sample_volume[:3] != img_size[:3]:
                raise ValueError(
                    f"fMRI spatial shape mismatch: manifest sample uses {sample_volume[:3]}, but fmri_model.img_size expects {img_size[:3]}. "
                    "Set data.fmri_spatial_strategy to interpolate or pad_or_crop if resizing is intended."
                )
            if temporal_strategy == "none" and sample_volume[3] != img_size[3]:
                raise ValueError(
                    f"fMRI time length mismatch: manifest sample uses {sample_volume[3]}, but fmri_model.img_size expects {img_size[3]}. "
                    "Set data.fmri_temporal_strategy to interpolate or pad_or_crop if resizing is intended."
                )

            patch_size = tuple(int(item) for item in fmri_cfg.get("patch_size", (6, 6, 6, 1)))
            if len(patch_size) != 4:
                raise ValueError(f"fmri_model.patch_size must be a 4-element sequence, got {patch_size}")
            if patch_size[3] != 1:
                raise ValueError(f"NeuroSTORM requires fmri_model.patch_size[3] == 1, got {patch_size}")
            for axis_size, patch in zip(img_size, patch_size):
                if patch <= 0:
                    raise ValueError(f"fmri_model.patch_size entries must be positive, got {patch_size}")
                if axis_size % patch != 0:
                    raise ValueError(f"fmri_model.img_size {img_size} must be divisible by patch_size {patch_size}")
        else:
            raise ValueError(f"Unsupported data.fmri_input_type: {fmri_input_type}")


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
        pretrain_objective = str(train_cfg.get("pretrain_objective", "shared_private")).strip().lower()
        eeg_shared_dim = int(eeg_cfg.get("shared_dim", train_cfg.get("projection_dim", 256)))
        eeg_private_dim = int(eeg_cfg.get("private_dim", eeg_shared_dim))
        eeg_band_power_dim = int(eeg_cfg.get("band_power_dim", 5))
        fmri_shared_dim = int(fmri_cfg.get("shared_dim", eeg_shared_dim))

        required_sections = ["train", "data", "eeg_model", "fmri_model"]
        for section_name in required_sections:
            if section_name not in self.raw:
                raise ValueError(f"Missing required config section: {section_name}")

        if pretrain_objective not in {"shared_private", "infonce", "barlow_twins"}:
            raise ValueError("train.pretrain_objective must be one of: shared_private, infonce, barlow_twins")
        if eeg_shared_dim <= 0 or eeg_private_dim <= 0:
            raise ValueError("eeg_model.shared_dim and eeg_model.private_dim must be positive")
        if pretrain_objective == "shared_private" and eeg_band_power_dim != 5:
            raise ValueError("eeg_model.band_power_dim must be 5 for the fixed EEG band-power target")
        if fmri_shared_dim != eeg_shared_dim:
            raise ValueError("fmri_model.shared_dim must match eeg_model.shared_dim for shared InfoNCE")

        # 对比学习使用单一 manifest；微调仍使用 train/val/test manifests。
        manifest_value = str(data_cfg.get("manifest_csv", data_cfg.get("train_manifest_csv", "")))
        manifest_path = root / manifest_value
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        root_dir_value = str(data_cfg.get("root_dir", "")).strip()
        root_dir = (root / root_dir_value) if root_dir_value else None
        if root_dir is not None and not root_dir.exists():
            raise FileNotFoundError(f"Dataset root_dir not found: {root_dir}")

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
            eeg_encoder_variant = str(finetune_cfg.get("eeg_encoder_variant", "shared_private")).strip().lower()
            if eeg_encoder_variant not in {"shared_private", "shared_only"}:
                raise ValueError("finetune.eeg_encoder_variant must be one of: shared_private, shared_only")
            classifier_mode = str(finetune_cfg.get("classifier_mode", "concat")).strip().lower()
            if classifier_mode not in {"shared", "private", "concat", "add"}:
                raise ValueError("finetune.classifier_mode must be one of: shared, private, concat, add")
            if eeg_encoder_variant == "shared_only" and classifier_mode != "shared":
                raise ValueError("finetune.eeg_encoder_variant=shared_only requires finetune.classifier_mode=shared")
            visualization_cfg = finetune_cfg.get("visualization", {}) or {}
            train_curve_cfg = visualization_cfg.get("train_curve", {}) or {}
            if "enabled" in train_curve_cfg and not isinstance(train_curve_cfg.get("enabled"), bool):
                raise ValueError("finetune.visualization.train_curve.enabled must be a boolean")
        for split_key in ["train_manifest_csv", "val_manifest_csv", "test_manifest_csv"]:
            split_value = data_cfg.get(split_key, "")
            split_path = "" if split_value is None else str(split_value).strip()
            if split_path:
                resolved = root / split_path
                if not resolved.exists():
                    raise FileNotFoundError(f"Manifest CSV not found: {resolved}")
        eeg_channel_target_manifest = str(data_cfg.get("eeg_channel_target_manifest", "")).strip()
        if eeg_channel_target_manifest:
            resolved = root / eeg_channel_target_manifest
            if not resolved.exists():
                raise FileNotFoundError(f"EEG channel target manifest not found: {resolved}")
            if "num_classes" not in finetune_cfg:
                raise ValueError("Missing finetune.num_classes in config")
            selection_metric = str(finetune_cfg.get("selection_metric", "accuracy")).strip().lower()
            if selection_metric not in {"accuracy", "acc", "macro_f1", "f1"}:
                raise ValueError("finetune.selection_metric must be one of: accuracy, acc, macro_f1, f1")
            contrastive_checkpoint = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()
            if contrastive_checkpoint:
                resolved = root / contrastive_checkpoint
                if not resolved.exists():
                    raise FileNotFoundError(f"Contrastive checkpoint not found: {resolved}")
            eeg_baseline_cfg = finetune_cfg.get("eeg_baseline", {}) or {}
            if bool(eeg_baseline_cfg.get("enabled", False)):
                model_name = str(eeg_baseline_cfg.get("model_name", "cbramod")).strip().lower()
                fusion = str(finetune_cfg.get("fusion", "eeg_only")).strip().lower()
                if fusion not in {"eeg_only", "fmri_only", "concat", "add"}:
                    raise ValueError("finetune.fusion must be one of: eeg_only, fmri_only, concat, add")
                valid_models = {
                        "traditional": {"svm", "eeg_deformer", "eegnet", "conformer", "tsception"},
                        "foundation": {"labram", "cbramod"},
                }
                inferred_category = next(
                    (name for name, models in valid_models.items() if model_name in models),
                    "",
                )
                if not inferred_category:
                    supported_models = sorted(valid_models["traditional"] | valid_models["foundation"])
                    raise ValueError(
                        f"Unsupported finetune.eeg_baseline.model_name='{model_name}'. "
                        f"Supported values: {supported_models}"
                    )
                raw_category = str(eeg_baseline_cfg.get("category", "")).strip().lower()
                if raw_category and raw_category not in {"traditional", "foundation"}:
                    raise ValueError("finetune.eeg_baseline.category must be one of: traditional, foundation")
                category = raw_category or inferred_category
                if category != inferred_category:
                    category = inferred_category
                if model_name not in valid_models[category]:
                    raise ValueError(
                        f"Unsupported finetune.eeg_baseline.model_name='{model_name}' for category='{category}'"
                    )
                if category == "traditional" and fusion != "eeg_only":
                    raise ValueError("Traditional EEG baselines only support finetune.fusion=eeg_only")
                if category == "foundation":
                    load_pretrained = bool(eeg_baseline_cfg.get("load_pretrained_weights", True))
                    checkpoint_path = str(eeg_baseline_cfg.get("checkpoint_path", "")).strip()
                    if load_pretrained and checkpoint_path:
                        resolved = root / checkpoint_path
                        if not resolved.exists():
                            raise FileNotFoundError(f"EEG baseline checkpoint not found: {resolved}")


        _validate_manifest_shapes(manifest_path=manifest_path, root_dir=root_dir, eeg_cfg=eeg_cfg, fmri_cfg=fmri_cfg, data_cfg=data_cfg)
