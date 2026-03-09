from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Align subject-packed ds002739 EEG channels by shared channel indices")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/ds002739_subject_packed_eeg2_fmri6"),
        help="Root directory containing manifest_all.csv and subjects/*.npz",
    )
    parser.add_argument(
        "--target-channels",
        type=int,
        default=None,
        help="Optional explicit target channel count. Defaults to the minimum channel count across all subject packs.",
    )
    return parser.parse_args()


def load_subject_channel_count(subject_path: Path) -> int:
    with np.load(subject_path, allow_pickle=False) as data:
        eeg = data["eeg"]
        if eeg.ndim != 4:
            raise ValueError(f"Expected packed EEG shape [N,C,S,P], got {eeg.shape} in {subject_path}")
        return int(eeg.shape[1])


def rewrite_subject_pack(subject_path: Path, target_channels: int) -> tuple[int, int]:
    with np.load(subject_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    eeg = np.asarray(payload["eeg"])
    original_channels = int(eeg.shape[1])
    if original_channels < target_channels:
        raise ValueError(f"Subject {subject_path} has only {original_channels} channels, smaller than target {target_channels}")

    if original_channels == target_channels:
        return original_channels, target_channels

    payload["eeg"] = np.asarray(eeg[:, :target_channels, :, :], dtype=eeg.dtype)
    np.savez(subject_path, **payload)
    return original_channels, target_channels


def refresh_manifest_shapes(manifest_path: Path, dataset_root: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    if "subject_path" not in manifest.columns:
        return

    eeg_shapes: list[str] = []
    fmri_shapes: list[str] = []
    label_shapes: list[str] = []
    sample_counts: list[int] = []

    for _, row in manifest.iterrows():
        subject_path = dataset_root / str(row["subject_path"])
        with np.load(subject_path, allow_pickle=False) as data:
            eeg = data["eeg"]
            fmri = data["fmri"]
            labels = data["labels"] if "labels" in data else None
            sample_counts.append(int(eeg.shape[0]))
            eeg_shapes.append("x".join(str(dim) for dim in eeg.shape))
            fmri_shapes.append("x".join(str(dim) for dim in fmri.shape))
            label_shapes.append("x".join(str(dim) for dim in labels.shape) if labels is not None else str(eeg.shape[0]))

    manifest["sample_count"] = sample_counts
    manifest["eeg_shape"] = eeg_shapes
    manifest["fmri_shape"] = fmri_shapes
    manifest["label_shape"] = label_shapes
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest_all.csv"
    subjects_dir = dataset_root / "subjects"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not subjects_dir.exists():
        raise FileNotFoundError(f"Subjects directory not found: {subjects_dir}")

    subject_paths = sorted(subjects_dir.glob("sub-*.npz"))
    if not subject_paths:
        raise RuntimeError(f"No subject packs found under {subjects_dir}")

    channel_counts = {path.name: load_subject_channel_count(path) for path in subject_paths}
    target_channels = int(args.target_channels) if args.target_channels is not None else min(channel_counts.values())
    if target_channels <= 0:
        raise ValueError(f"Invalid target channel count: {target_channels}")

    print(f"Channel counts before alignment: {channel_counts}")
    print(f"Using shared channel indices: [0:{target_channels}]")

    changed = 0
    for subject_path in subject_paths:
        original_channels, new_channels = rewrite_subject_pack(subject_path, target_channels=target_channels)
        if original_channels != new_channels:
            changed += 1

    refresh_manifest_shapes(manifest_path, dataset_root)
    for split_manifest in dataset_root.glob("**/manifest_*.csv"):
        refresh_manifest_shapes(split_manifest, dataset_root)

    print(f"Aligned EEG channels to {target_channels}. Updated {changed} subject packs.")
    print(f"Updated manifests under {dataset_root}")


if __name__ == "__main__":
    main()