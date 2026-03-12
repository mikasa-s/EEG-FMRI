from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shutil
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from scipy.ndimage import zoom
from scipy.signal import resample

sys.path.append(str(Path(__file__).resolve().parent.parent))


def add_dataset_io_args(parser: argparse.ArgumentParser, ds_root_help: str, output_root_help: str) -> None:
    parser.add_argument("--ds-root", type=Path, required=True, help=ds_root_help)
    parser.add_argument("--output-root", type=Path, required=True, help=output_root_help)


def add_subject_args(parser: argparse.ArgumentParser, subject_example: str) -> None:
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=f"Optional subject IDs such as {subject_example}. Defaults to all subjects found under ds-root.",
    )


def add_atlas_args(
    parser: argparse.ArgumentParser,
    default_atlas_name: str = "schaefer",
    atlas_choices: Sequence[str] = ("schaefer",),
    default_n_rois: int = 400,
) -> None:
    parser.add_argument(
        "--atlas-labels-img",
        type=Path,
        default=None,
        help="Optional path to a labels atlas image in MNI space. If omitted, Schaefer atlas is fetched.",
    )
    parser.add_argument("--atlas-name", default=default_atlas_name, choices=list(atlas_choices), help="Built-in atlas preset.")
    parser.add_argument("--n-rois", type=int, default=default_n_rois, help="ROI count for the built-in Schaefer atlas.")


def add_common_fmri_args(
    parser: argparse.ArgumentParser,
    *,
    default_fmri_mode: str,
    tr_help: str,
    standardize_help: str,
    fmri_max_shape_help: str,
) -> None:
    parser.add_argument("--tr", type=float, default=2.0, help=tr_help)
    parser.add_argument(
        "--fmri-mode",
        default=default_fmri_mode,
        choices=["roi", "volume"],
        help="roi saves [ROI,T]; volume saves normalized 4D windows [H,W,D,T].",
    )
    parser.add_argument(
        "--fmri-voxel-size",
        nargs=3,
        type=float,
        default=[2.0, 2.0, 2.0],
        help="Target spatial voxel size in mm for volume-mode fMRI preprocessing.",
    )
    parser.add_argument(
        "--fmri-max-shape",
        nargs=3,
        type=int,
        default=[48, 48, 48],
        help=fmri_max_shape_help,
    )
    parser.add_argument(
        "--fmri-float16",
        action="store_true",
        help="Save preprocessed volume-mode fMRI windows as float16 instead of float32.",
    )
    parser.add_argument(
        "--standardize-fmri",
        action="store_true",
        help=standardize_help,
    )


def add_fmri_roi_resample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fmri-target-rois",
        type=int,
        default=None,
        help="Optional target ROI count for fMRI. Disabled by default because ROI interpolation changes the atlas semantics.",
    )
    parser.add_argument(
        "--fmri-target-t",
        type=int,
        default=None,
        help="Optional target time length for fMRI. Disabled by default because temporal interpolation changes the raw TR grid.",
    )
    parser.add_argument(
        "--allow-fmri-roi-resample",
        action="store_true",
        help="Explicitly allow Fourier resampling on the ROI axis. Use only for debugging, not for raw-faithful exports.",
    )
    parser.add_argument(
        "--allow-fmri-time-resample",
        action="store_true",
        help="Explicitly allow Fourier resampling on the time axis. Use only for debugging, not for raw-faithful exports.",
    )


def add_eeg_patch_args(
    parser: argparse.ArgumentParser,
    *,
    default_eeg_mode: str,
    default_seq_len: int | None,
    default_patch_len: int | None,
    seq_len_help: str,
    patch_len_help: str,
) -> None:
    parser.add_argument(
        "--eeg-mode",
        default=default_eeg_mode,
        choices=["continuous", "patched"],
        help="continuous saves [C,T]; patched saves [C,S,P] for direct use by the default dataset.",
    )
    parser.add_argument("--eeg-seq-len", type=int, default=default_seq_len, help=seq_len_help)
    parser.add_argument("--eeg-patch-len", type=int, default=default_patch_len, help=patch_len_help)


def add_subject_packing_and_split_args(
    parser: argparse.ArgumentParser,
    *,
    pack_help: str,
    split_help: str,
    train_subjects: int,
    val_subjects: int,
    test_subjects: int,
) -> None:
    parser.add_argument(
        "--pack-subject-files",
        action="store_true",
        help=pack_help,
    )
    parser.add_argument(
        "--split-mode",
        default="loso",
        choices=["none", "subject", "loso"],
        help=split_help,
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated split manifests. Defaults to <output-root>/splits_subjectwise or <output-root>/loso_subjectwise.",
    )
    parser.add_argument("--train-subjects", type=int, default=train_subjects, help="Number of subjects in train split when --split-mode=subject.")
    parser.add_argument("--val-subjects", type=int, default=val_subjects, help="Number of subjects in val split or LOSO validation subjects.")
    parser.add_argument("--test-subjects", type=int, default=test_subjects, help="Number of subjects in test split when --split-mode=subject.")


def add_training_ready_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-ready",
        dest="training_ready",
        action="store_true",
        help="Apply training-time normalization during preprocessing and mark manifests so dataset skips normalization and fMRI shape adaptation.",
    )
    parser.add_argument(
        "--no-training-ready",
        dest="training_ready",
        action="store_false",
        help="Disable training-ready preprocessing and keep normalization/fMRI shape adaptation in dataset initialization.",
    )
    parser.set_defaults(training_ready=True)


def find_subjects(ds_root: Path, requested_subjects: list[str] | None) -> list[str]:
    if requested_subjects:
        return requested_subjects
    return sorted(path.name for path in ds_root.glob("sub-*") if path.is_dir())


def get_atlas_labels_img(atlas_labels_img: Path | None, atlas_cache_dir: Path, n_rois: int) -> str:
    if atlas_labels_img is not None:
        return str(atlas_labels_img)
    atlas_cache_dir.mkdir(parents=True, exist_ok=True)
    atlas = fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(atlas_cache_dir),
    )
    return str(atlas.maps)


def resample_fmri_if_needed(
    series: np.ndarray,
    fmri_target_rois: int | None,
    fmri_target_t: int | None,
    allow_roi_resample: bool,
    allow_time_resample: bool,
) -> np.ndarray:
    if fmri_target_rois is not None and series.shape[0] != fmri_target_rois:
        if not allow_roi_resample:
            raise ValueError(
                f"Requested fmri_target_rois={fmri_target_rois}, but extracted ROI count is {series.shape[0]}. "
                "Provide a real atlas with the desired ROI count instead of interpolating, or pass --allow-fmri-roi-resample to override."
            )
        series = resample(series, fmri_target_rois, axis=0).astype(np.float32)
    if fmri_target_t is not None and series.shape[1] != fmri_target_t:
        if not allow_time_resample:
            raise ValueError(
                f"Requested fmri_target_t={fmri_target_t}, but the block contains {series.shape[1]} time points. "
                "Keep the native TR grid for raw-faithful exports, or pass --allow-fmri-time-resample to override."
            )
        series = resample(series, fmri_target_t, axis=1).astype(np.float32)
    return series.astype(np.float32)


def extract_roi_timeseries(
    fmri_nii_path: Path,
    labels_img: str,
    tr: float,
    standardize_fmri: bool,
    discard_initial_trs: int = 0,
    fmri_target_t: int | None = None,
    allow_time_resample: bool = False,
    include_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
    img = nib.load(str(fmri_nii_path))
    original_shape = tuple(int(dim) for dim in img.shape)
    voxel_size = tuple(float(dim) for dim in img.header.get_zooms())
    if discard_initial_trs > 0:
        img = nib.Nifti1Image(img.get_fdata(dtype=np.float32)[..., discard_initial_trs:], img.affine, img.header)
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=standardize_fmri, detrend=True, t_r=tr)
    series = masker.fit_transform(img).T.astype(np.float32)
    series = resample_fmri_if_needed(
        series,
        fmri_target_rois=None,
        fmri_target_t=fmri_target_t,
        allow_roi_resample=False,
        allow_time_resample=allow_time_resample,
    )
    if include_metadata:
        return series, original_shape, voxel_size
    return series


def load_bold_volume(
    fmri_nii_path: Path,
    discard_initial_trs: int = 0,
    include_metadata: bool = False,
) -> tuple[np.ndarray, tuple[float, ...]] | tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
    img = nib.load(str(fmri_nii_path))
    volume = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    original_shape = tuple(int(dim) for dim in img.shape)
    voxel_size = tuple(float(dim) for dim in img.header.get_zooms())
    if discard_initial_trs > 0:
        volume = volume[..., discard_initial_trs:]
    if include_metadata:
        return volume, original_shape, voxel_size
    return volume, voxel_size


def spatial_resample_volume(data: np.ndarray, voxel_size: Sequence[float], target_voxel_size: Sequence[float]) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    scale_factors = tuple(float(current) / float(target) for current, target in zip(voxel_size[:3], target_voxel_size[:3]))
    return zoom(data, zoom=(scale_factors[0], scale_factors[1], scale_factors[2], 1.0), order=1).astype(np.float32)


def temporal_resample_volume(data: np.ndarray, source_tr: float, target_tr: float) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    if abs(source_tr - target_tr) < 1e-6:
        return data.astype(np.float32)
    target_t = max(1, int(round(data.shape[3] * float(source_tr) / float(target_tr))))
    return zoom(data, zoom=(1.0, 1.0, 1.0, target_t / max(data.shape[3], 1)), order=1).astype(np.float32)


def center_crop_spatial_max(data: np.ndarray, max_shape: Sequence[int]) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    slices: list[slice] = []
    for axis, max_size in enumerate(max_shape[:3]):
        current_size = data.shape[axis]
        if current_size > int(max_size):
            start = (current_size - int(max_size)) // 2
            end = start + int(max_size)
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, current_size))
    slices.append(slice(None))
    return data[tuple(slices)].astype(np.float32)


def preprocess_fmri_volume(
    data: np.ndarray,
    voxel_size: Sequence[float],
    source_tr: float,
    target_voxel_size: Sequence[float],
    target_tr: float,
    max_shape: Sequence[int],
    use_float16: bool = False,
) -> np.ndarray:
    output = spatial_resample_volume(data, voxel_size=voxel_size, target_voxel_size=target_voxel_size)
    output = temporal_resample_volume(output, source_tr=source_tr, target_tr=target_tr)
    output = center_crop_spatial_max(output, max_shape=max_shape)
    if use_float16:
        return output.astype(np.float16)
    return output.astype(np.float32)


def zscore_array_sample(data: np.ndarray) -> np.ndarray:
    output = np.asarray(data, dtype=np.float32)
    mean = float(output.mean())
    std = float(output.std()) + 1e-6
    return ((output - mean) / std).astype(np.float32)


def zscore_nonzero_volume_sample(data: np.ndarray) -> np.ndarray:
    output = np.asarray(data, dtype=np.float32)
    mask = np.abs(output) > 1e-8
    if not np.any(mask):
        return output.astype(np.float32)
    mean = float(output[mask].mean())
    std = float(output[mask].std()) + 1e-6
    normalized = np.array(output, copy=True)
    normalized[mask] = (normalized[mask] - mean) / std
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def prepare_training_ready_eeg(data: np.ndarray, enabled: bool) -> np.ndarray:
    output = np.asarray(data, dtype=np.float32)
    if not enabled:
        return output.astype(np.float32, copy=False)
    return zscore_array_sample(output)


def prepare_training_ready_fmri(data: np.ndarray, fmri_mode: str, enabled: bool) -> np.ndarray:
    output = np.asarray(data, dtype=np.float32)
    if not enabled:
        return output.astype(np.float32, copy=False)
    if str(fmri_mode).strip().lower() == "volume":
        return zscore_nonzero_volume_sample(output)
    return zscore_array_sample(output)


def stack_subject_samples(samples: list[np.ndarray], name: str) -> np.ndarray:
    if not samples:
        raise ValueError(f"No {name} samples available for subject packing.")
    first_shape = tuple(samples[0].shape)
    for sample in samples[1:]:
        if tuple(sample.shape) != first_shape:
            raise ValueError(f"Cannot pack subject-level {name} arrays with inconsistent shapes: {first_shape} vs {tuple(sample.shape)}")
    return np.stack(samples, axis=0)


def write_subject_memmap_pack(pack_dir: Path, arrays: dict[str, np.ndarray]) -> Path:
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        np_array = np.asarray(array)
        out_path = pack_dir / f"{name}.npy"
        memmap = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np_array.dtype,
            shape=np_array.shape,
        )
        memmap[...] = np_array
        memmap.flush()
        del memmap
        metadata[name] = {
            "dtype": str(np_array.dtype),
            "shape": [int(dim) for dim in np_array.shape],
        }

    with open(pack_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"format": "subject_memmap_v1", "arrays": metadata}, handle, ensure_ascii=False, indent=2)

    return pack_dir


def format_label_distribution(split_df: pd.DataFrame) -> dict[int, int]:
    if "label" not in split_df.columns:
        return {}
    return {int(key): int(value) for key, value in split_df["label"].value_counts().sort_index().items()}


def choose_val_subjects(subjects: list[str], held_out_index: int, val_subjects: int) -> list[str]:
    rotated = subjects[held_out_index + 1 :] + subjects[:held_out_index]
    return rotated[:val_subjects]


def write_subject_splits(
    manifest_path: Path,
    output_dir: Path,
    train_subjects: int,
    val_subjects: int,
    test_subjects: int,
) -> Path:
    manifest = pd.read_csv(manifest_path)
    if "subject" not in manifest.columns:
        raise ValueError("Manifest must contain a 'subject' column for subject-wise splitting")

    subjects = sorted(manifest["subject"].dropna().unique().tolist())
    requested = train_subjects + val_subjects + test_subjects
    if requested > len(subjects):
        raise ValueError(
            f"Requested {requested} subjects across splits, but manifest only contains {len(subjects)} subjects"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_subject_ids = subjects[:train_subjects]
    val_subject_ids = subjects[train_subjects : train_subjects + val_subjects]
    test_subject_ids = subjects[
        train_subjects + val_subjects : train_subjects + val_subjects + test_subjects
    ]

    split_map = {
        "train": train_subject_ids,
        "val": val_subject_ids,
        "test": test_subject_ids,
    }

    summary_rows: list[dict[str, object]] = []
    for split_name, split_subjects in split_map.items():
        split_df = manifest[manifest["subject"].isin(split_subjects)].copy()
        split_df.to_csv(output_dir / f"manifest_{split_name}.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "subjects": ",".join(split_subjects),
                "num_subjects": len(split_subjects),
                "num_samples": int(len(split_df)),
                "label_distribution": format_label_distribution(split_df),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False)
    return output_dir


def write_loso_splits(manifest_path: Path, output_dir: Path, val_subjects: int) -> Path:
    manifest = pd.read_csv(manifest_path)
    if "subject" not in manifest.columns:
        raise ValueError("Manifest must contain a 'subject' column for LOSO splitting")

    subjects = sorted(manifest["subject"].dropna().unique().tolist())
    if val_subjects <= 0:
        raise ValueError("val_subjects must be positive")
    if val_subjects >= len(subjects):
        raise ValueError("val_subjects must be smaller than the total number of subjects")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for held_out_index, test_subject in enumerate(subjects):
        val_subject_ids = choose_val_subjects(subjects, held_out_index, val_subjects)
        train_subject_ids = [subject for subject in subjects if subject not in {test_subject, *val_subject_ids}]

        fold_name = f"fold_{test_subject}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        split_specs = {
            "train": train_subject_ids,
            "val": val_subject_ids,
            "test": [test_subject],
        }

        for split_name, split_subjects in split_specs.items():
            split_df = manifest[manifest["subject"].isin(split_subjects)].copy()
            split_df.to_csv(fold_dir / f"manifest_{split_name}.csv", index=False)
            summary_rows.append(
                {
                    "fold": fold_name,
                    "split": split_name,
                    "subjects": ",".join(split_subjects),
                    "num_subjects": len(split_subjects),
                    "num_samples": int(len(split_df)),
                    "label_distribution": format_label_distribution(split_df),
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "loso_summary.csv", index=False)
    return output_dir
