from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm


SCENE_BINARY_LABELS = {
    "dot_stim_validtrials": (0, "motion_stimulus"),
    "rating_stim_validtrials": (1, "confidence_rating"),
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    subject: str
    task: str
    run: str
    trial_type: str
    eeg_path: str
    fmri_path: str
    label: int
    label_name: str
    eeg_shape: str
    fmri_shape: str


@dataclass(frozen=True)
class RunSummary:
    subject: str
    run: str
    bold_shape: str
    voxel_size: str
    roi_shape: str
    eeg_shape: str
    eeg_fmri_offset_sec: float
    event_counts: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds002739 EEG/fMRI pairs for binary scene classification.")
    parser.add_argument("--ds-root", type=Path, required=True, help="Path to ds002739 root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for exported arrays and manifests.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs such as sub-01. Defaults to all subjects found under ds-root.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional run IDs such as run-01 run-02. Defaults to all runs found per subject.",
    )
    parser.add_argument(
        "--event-types",
        nargs=2,
        default=["dot_stim_validtrials", "rating_stim_validtrials"],
        help="Exactly two real event types to export for binary classification.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Fixed window length in seconds extracted after each event onset for both EEG and fMRI.",
    )
    parser.add_argument(
        "--atlas-labels-img",
        type=Path,
        default=None,
        help="Optional path to a labels atlas image in MNI space. If omitted, Schaefer atlas is fetched.",
    )
    parser.add_argument("--atlas-name", default="schaefer", choices=["schaefer"], help="Built-in atlas preset.")
    parser.add_argument("--n-rois", type=int, default=400, help="ROI count for the built-in Schaefer atlas.")
    parser.add_argument("--tr", type=float, default=2.0, help="fMRI repetition time in seconds.")
    parser.add_argument(
        "--eeg-mode",
        default="patched",
        choices=["continuous", "patched"],
        help="continuous saves [C,T]; patched saves [C,S,P] for direct use by the default dataset.",
    )
    parser.add_argument(
        "--eeg-seq-len",
        type=int,
        default=None,
        help="Optional EEG patch count when eeg-mode=patched. Defaults to rounded window seconds.",
    )
    parser.add_argument(
        "--eeg-patch-len",
        type=int,
        default=None,
        help="Optional EEG patch length when eeg-mode=patched. Defaults to the EEG sampling rate.",
    )
    parser.add_argument(
        "--standardize-fmri",
        action="store_true",
        help="Apply standardization inside the ROI masker.",
    )
    return parser.parse_args()


def find_subjects(ds_root: Path, requested_subjects: list[str] | None) -> list[str]:
    if requested_subjects:
        return requested_subjects
    return sorted(path.name for path in ds_root.glob("sub-*") if path.is_dir())


def get_atlas_labels_img(args: argparse.Namespace, atlas_cache_dir: Path) -> str:
    if args.atlas_labels_img is not None:
        return str(args.atlas_labels_img)
    atlas_cache_dir.mkdir(parents=True, exist_ok=True)
    atlas = fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(atlas_cache_dir),
    )
    return str(atlas.maps)


def load_mat_payload(path: Path) -> dict:
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def load_eeg_data(eeg_mat_path: Path) -> tuple[np.ndarray, float]:
    payload = load_mat_payload(eeg_mat_path)
    eeg_struct = payload["EEGdata"]
    eeg = np.asarray(eeg_struct.Y, dtype=np.float32)
    fs = float(payload.get("fs", getattr(eeg_struct, "fs", 1000.0)))
    if eeg.ndim != 2:
        raise ValueError(f"Expected EEG data with shape [channels, samples], got {eeg.shape} from {eeg_mat_path}")
    return eeg, fs


def load_eeg_events(events_mat_path: Path) -> dict[str, np.ndarray]:
    payload = load_mat_payload(events_mat_path)
    return {key: np.asarray(value).reshape(-1) for key, value in payload.items() if not key.startswith("__")}


def extract_roi_timeseries(
    fmri_nii_path: Path,
    labels_img: str,
    tr: float,
    standardize_fmri: bool,
) -> tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
    img = nib.load(str(fmri_nii_path))
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=standardize_fmri, detrend=True, t_r=tr)
    series = masker.fit_transform(img).T.astype(np.float32)
    return series, tuple(int(dim) for dim in img.shape), tuple(float(dim) for dim in img.header.get_zooms())


def get_run_ids(func_dir: Path, requested_runs: list[str] | None) -> list[str]:
    if requested_runs is not None:
        return requested_runs
    run_ids = []
    for path in sorted(func_dir.glob("*_events.tsv")):
        name = path.name
        marker = "_run-"
        start = name.find(marker)
        if start < 0:
            continue
        run_ids.append(name[start + 1 : start + 7])
    return sorted(set(run_ids))


def load_fmri_events(events_tsv_path: Path, event_types: list[str]) -> pd.DataFrame:
    events = pd.read_csv(events_tsv_path, sep="\t")
    required = {"onset", "duration", "trial_type"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing required columns {required} in {events_tsv_path}")
    events = events.dropna(subset=["onset", "duration", "trial_type"]).copy()
    return events[events["trial_type"].isin(event_types)].reset_index(drop=True)


def estimate_eeg_fmri_offset_sec(eeg_events: dict[str, np.ndarray], fmri_events: pd.DataFrame) -> float:
    if "tstim" not in eeg_events:
        raise ValueError("EEG events file is missing tstim, cannot estimate EEG-fMRI alignment.")
    eeg_tstim = np.asarray(eeg_events["tstim"], dtype=np.float64).reshape(-1) / 1000.0
    fmri_tstim = fmri_events.loc[fmri_events["trial_type"] == "dot_stim_validtrials", "onset"].to_numpy(dtype=np.float64)
    if len(eeg_tstim) == 0 or len(fmri_tstim) == 0:
        raise ValueError("Cannot estimate alignment offset because dot stimulus events are missing.")
    count = min(len(eeg_tstim), len(fmri_tstim))
    diff = eeg_tstim[:count] - fmri_tstim[:count]
    return float(np.median(diff))


def slice_eeg_window(data: np.ndarray, sfreq: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    end = start + length
    if start < 0 or end > data.shape[1]:
        raise ValueError(f"EEG slice out of range: start={start}, end={end}, total={data.shape[1]}")
    return data[:, start:end].astype(np.float32)


def slice_fmri_window(series: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = max(1, int(round(duration_sec / tr)))
    end = start + length
    if start < 0 or end > series.shape[1]:
        raise ValueError(f"fMRI slice out of range: start={start}, end={end}, total={series.shape[1]}")
    return series[:, start:end].astype(np.float32)


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    if data.shape[1] != target_len:
        data = resample(data, target_len, axis=1).astype(np.float32)
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def resolve_eeg_patch_params(sfreq: float, args: argparse.Namespace) -> tuple[int, int]:
    patch_len = args.eeg_patch_len if args.eeg_patch_len is not None else int(round(sfreq))
    seq_len = args.eeg_seq_len if args.eeg_seq_len is not None else max(1, int(round(args.window_sec)))
    if patch_len <= 0 or seq_len <= 0:
        raise ValueError(f"Invalid EEG patch params: seq_len={seq_len}, patch_len={patch_len}")
    return seq_len, patch_len


def build_sample_record(
    sample_id: str,
    subject: str,
    run: str,
    trial_type: str,
    eeg_rel_path: Path,
    fmri_rel_path: Path,
    label: int,
    label_name: str,
    eeg: np.ndarray,
    fmri: np.ndarray,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        subject=subject,
        task="main",
        run=run,
        trial_type=trial_type,
        eeg_path=eeg_rel_path.as_posix(),
        fmri_path=fmri_rel_path.as_posix(),
        label=label,
        label_name=label_name,
        eeg_shape="x".join(str(dim) for dim in eeg.shape),
        fmri_shape="x".join(str(dim) for dim in fmri.shape),
    )


def iter_subject_runs(subjects: Iterable[str], ds_root: Path, requested_runs: list[str] | None) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        func_dir = ds_root / subject / "func"
        for run in get_run_ids(func_dir, requested_runs):
            yield subject, run


def main() -> None:
    args = parse_args()
    event_types = [event_type.strip() for event_type in args.event_types]
    if len(event_types) != 2:
        raise ValueError("--event-types must contain exactly two entries for binary classification.")
    unsupported = [event_type for event_type in event_types if event_type not in SCENE_BINARY_LABELS]
    if unsupported:
        raise ValueError(f"Unsupported event types for scene binary labeling: {unsupported}")
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")

    ds_root = args.ds_root.resolve()
    out_root = args.output_root.resolve()
    eeg_out_dir = out_root / "eeg"
    fmri_out_dir = out_root / "fmri"
    atlas_cache_dir = out_root / "atlas_cache"
    eeg_out_dir.mkdir(parents=True, exist_ok=True)
    fmri_out_dir.mkdir(parents=True, exist_ok=True)

    labels_img = get_atlas_labels_img(args, atlas_cache_dir)
    subjects = find_subjects(ds_root, args.subjects)
    records: list[SampleRecord] = []
    summaries: list[RunSummary] = []

    for subject, run in tqdm(list(iter_subject_runs(subjects, ds_root, args.runs)), desc="Preparing ds002739"):
        func_dir = ds_root / subject / "func"
        eeg_dir = ds_root / subject / "EEG"
        bold_path = func_dir / f"{subject}_task-main_{run}_bold.nii.gz"
        fmri_events_path = func_dir / f"{subject}_task-main_{run}_events.tsv"
        eeg_data_path = eeg_dir / f"EEG_data_{subject}_{run}.mat"
        eeg_events_path = eeg_dir / f"EEG_events_{subject}_{run}.mat"
        if not all(path.exists() for path in [bold_path, fmri_events_path, eeg_data_path, eeg_events_path]):
            continue

        eeg_data, sfreq = load_eeg_data(eeg_data_path)
        eeg_events = load_eeg_events(eeg_events_path)
        fmri_events = load_fmri_events(fmri_events_path, event_types=event_types)
        if fmri_events.empty:
            continue

        fmri_series, bold_shape, voxel_size = extract_roi_timeseries(
            fmri_nii_path=bold_path,
            labels_img=labels_img,
            tr=args.tr,
            standardize_fmri=args.standardize_fmri,
        )
        eeg_fmri_offset_sec = estimate_eeg_fmri_offset_sec(eeg_events=eeg_events, fmri_events=fmri_events)
        eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(sfreq=sfreq, args=args)

        event_counts = fmri_events["trial_type"].value_counts().sort_index().to_dict()
        summaries.append(
            RunSummary(
                subject=subject,
                run=run,
                bold_shape="x".join(str(dim) for dim in bold_shape),
                voxel_size="x".join(str(dim) for dim in voxel_size),
                roi_shape="x".join(str(dim) for dim in fmri_series.shape),
                eeg_shape="x".join(str(dim) for dim in eeg_data.shape),
                eeg_fmri_offset_sec=eeg_fmri_offset_sec,
                event_counts=str(event_counts),
            )
        )

        for event_index, row in fmri_events.reset_index(drop=True).iterrows():
            trial_type = str(row["trial_type"]).strip()
            label, label_name = SCENE_BINARY_LABELS[trial_type]
            fmri_onset_sec = float(row["onset"])
            eeg_onset_sec = fmri_onset_sec + eeg_fmri_offset_sec

            eeg_window = slice_eeg_window(eeg_data, sfreq=sfreq, start_sec=eeg_onset_sec, duration_sec=args.window_sec)
            if args.eeg_mode == "patched":
                eeg_window = maybe_patch_eeg(eeg_window, seq_len=eeg_seq_len, patch_len=eeg_patch_len)

            fmri_window = slice_fmri_window(fmri_series, tr=args.tr, start_sec=fmri_onset_sec, duration_sec=args.window_sec)

            sample_id = f"{subject}_{run}_{trial_type}_{event_index:03d}"
            eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
            fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
            np.save(eeg_out_path, eeg_window)
            np.save(fmri_out_path, fmri_window)

            records.append(
                build_sample_record(
                    sample_id=sample_id,
                    subject=subject,
                    run=run,
                    trial_type=trial_type,
                    eeg_rel_path=eeg_out_path.relative_to(out_root),
                    fmri_rel_path=fmri_out_path.relative_to(out_root),
                    label=label,
                    label_name=label_name,
                    eeg=eeg_window,
                    fmri=fmri_window,
                )
            )

    if not records:
        raise RuntimeError("No samples were exported. Check subject IDs, run IDs, and input paths.")

    pd.DataFrame(record.__dict__ for record in records).to_csv(out_root / "manifest_all.csv", index=False)
    pd.DataFrame(summary.__dict__ for summary in summaries).to_csv(out_root / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()