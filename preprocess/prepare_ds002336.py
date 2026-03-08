from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import resample
from tqdm import tqdm


TASK_LABELS = {
    "motorloc": 0,
    "MIpre": 1,
    "MIpost": 2,
    "eegNF": 3,
    "fmriNF": 4,
    "eegfmriNF": 5,
}

TASK_DURATIONS_SEC = {
    "motorloc": 320,
    "MIpre": 200,
    "MIpost": 200,
    "eegNF": 400,
    "fmriNF": 400,
    "eegfmriNF": 400,
}

TRIAL_TYPE_BINARY_LABELS = {
    "rest": 0,
    "task-me": 1,
    "task-mi": 1,
    "task-nf": 1,
}

TRIAL_TYPE_LABEL_NAMES = {
    0: "non_motor",
    1: "motor",
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    subject: str
    task: str
    trial_type: str
    eeg_path: str
    fmri_path: str
    label: int
    label_name: str
    eeg_shape: str
    fmri_shape: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds002336 EEG/fMRI pairs for this repository.")
    parser.add_argument("--ds-root", type=Path, required=True, help="Path to ds002336 root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for converted arrays and manifests.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"],
        choices=list(TASK_LABELS.keys()),
        help="Tasks to export.",
    )
    parser.add_argument(
        "--sample-mode",
        default="run",
        choices=["run", "block"],
        help="run exports one sample per run; block exports one sample per rest/task block.",
    )
    parser.add_argument(
        "--label-mode",
        default="task",
        choices=["task", "binary_rest_task"],
        help="task uses task categories as labels; binary_rest_task uses non_motor=0 and motor=1.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs such as sub-xp101. Defaults to all subjects found under ds-root.",
    )
    parser.add_argument(
        "--atlas-labels-img",
        type=Path,
        default=None,
        help="Optional path to a labels atlas image in MNI space. If omitted, Schaefer atlas is fetched.",
    )
    parser.add_argument("--atlas-name", default="schaefer", choices=["schaefer"], help="Built-in atlas preset.")
    parser.add_argument("--n-rois", type=int, default=400, help="ROI count for the built-in Schaefer atlas.")
    parser.add_argument(
        "--fmri-target-rois",
        type=int,
        default=None,
        help="Optional target ROI count for fMRI. Uses Fourier resampling when set.",
    )
    parser.add_argument("--tr", type=float, default=2.0, help="fMRI repetition time.")
    parser.add_argument("--discard-initial-trs", type=int, default=2, help="Initial BOLD volumes to discard.")
    parser.add_argument(
        "--protocol-offset-sec",
        type=float,
        default=2.0,
        help="Seconds between fMRI acquisition start and protocol start in task events files.",
    )
    parser.add_argument(
        "--fmri-target-t",
        type=int,
        default=None,
        help="Optional target time length for fMRI. Uses Fourier resampling when set.",
    )
    parser.add_argument(
        "--eeg-mode",
        default="continuous",
        choices=["continuous", "patched"],
        help="continuous saves [C,T]; patched saves [C,S,P] for direct use by the default dataset.",
    )
    parser.add_argument("--eeg-seq-len", type=int, default=30, help="EEG patch count when eeg-mode=patched.")
    parser.add_argument("--eeg-patch-len", type=int, default=200, help="EEG patch length when eeg-mode=patched.")
    parser.add_argument(
        "--drop-ecg",
        action="store_true",
        help="Drop ECG and other non-EEG channels when reading the preprocessed BrainVision files.",
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


def load_eeg(eeg_vhdr_path: Path, drop_ecg: bool) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_brainvision(str(eeg_vhdr_path), preload=True, verbose="ERROR")
    if drop_ecg:
        raw.pick_types(eeg=True, ecg=False, misc=False, stim=False)
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"])


def crop_eeg_to_task(data: np.ndarray, task: str, sfreq: float) -> np.ndarray:
    duration_sec = TASK_DURATIONS_SEC[task]
    target_samples = int(round(duration_sec * sfreq))
    if data.shape[1] < target_samples:
        raise ValueError(f"EEG samples shorter than expected for task {task}: {data.shape[1]} < {target_samples}")
    return data[:, :target_samples]


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    if data.shape[1] != target_len:
        data = resample(data, target_len, axis=1)
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def resample_fmri_if_needed(
    series: np.ndarray,
    fmri_target_rois: int | None,
    fmri_target_t: int | None,
) -> np.ndarray:
    if fmri_target_rois is not None and series.shape[0] != fmri_target_rois:
        series = resample(series, fmri_target_rois, axis=0).astype(np.float32)
    if fmri_target_t is not None and series.shape[1] != fmri_target_t:
        series = resample(series, fmri_target_t, axis=1).astype(np.float32)
    return series.astype(np.float32)


def extract_roi_timeseries(
    fmri_nii_path: Path,
    labels_img: str,
    tr: float,
    discard_initial_trs: int,
    standardize_fmri: bool,
    fmri_target_t: int | None,
) -> np.ndarray:
    img = nib.load(str(fmri_nii_path))
    if discard_initial_trs > 0:
        img = nib.Nifti1Image(img.get_fdata(dtype=np.float32)[..., discard_initial_trs:], img.affine, img.header)
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=standardize_fmri, detrend=True, t_r=tr)
    series = masker.fit_transform(img)
    series = series.T.astype(np.float32)
    return resample_fmri_if_needed(series, None, fmri_target_t)


def load_task_events(ds_root: Path, task: str) -> pd.DataFrame:
    events_path = ds_root / f"task-{task}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(f"Task events TSV not found: {events_path}")
    return pd.read_csv(events_path, sep="\t").dropna(subset=["onset", "duration", "trial_type"])


def slice_eeg_block(data: np.ndarray, sfreq: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    end = start + length
    if start < 0 or end > data.shape[1]:
        raise ValueError(f"EEG block slice out of range: start={start}, end={end}, total={data.shape[1]}")
    return data[:, start:end].astype(np.float32)


def slice_fmri_block(series: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    if start < 0 or end > series.shape[1]:
        raise ValueError(f"fMRI block slice out of range: start={start}, end={end}, total={series.shape[1]}")
    return series[:, start:end].astype(np.float32)


def resolve_binary_label(trial_type: str) -> tuple[int, str]:
    normalized = trial_type.strip().lower()
    if normalized not in TRIAL_TYPE_BINARY_LABELS:
        raise ValueError(f"Unsupported trial_type for binary labeling: {trial_type}")
    label = TRIAL_TYPE_BINARY_LABELS[normalized]
    return label, TRIAL_TYPE_LABEL_NAMES[label]


def build_sample_record(
    sample_id: str,
    subject: str,
    task: str,
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
        task=task,
        trial_type=trial_type,
        eeg_path=eeg_rel_path.as_posix(),
        fmri_path=fmri_rel_path.as_posix(),
        label=label,
        label_name=label_name,
        eeg_shape="x".join(str(dim) for dim in eeg.shape),
        fmri_shape="x".join(str(dim) for dim in fmri.shape),
    )


def iter_subject_task_pairs(subjects: Iterable[str], tasks: Iterable[str]) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        for task in tasks:
            yield subject, task


def main() -> None:
    args = parse_args()
    if args.label_mode == "binary_rest_task" and args.sample_mode != "block":
        raise ValueError("binary_rest_task requires --sample-mode block because each run contains both rest and task blocks.")

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

    for subject, task in tqdm(list(iter_subject_task_pairs(subjects, args.tasks)), desc="Preparing ds002336"):
        eeg_vhdr = ds_root / "derivatives" / subject / "eeg_pp" / f"{subject}_task-{task}_eeg_pp.vhdr"
        fmri_nii = ds_root / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"
        if not eeg_vhdr.exists() or not fmri_nii.exists():
            continue

        eeg, sfreq = load_eeg(eeg_vhdr, drop_ecg=args.drop_ecg)
        if args.sample_mode == "run":
            eeg = crop_eeg_to_task(eeg, task, sfreq=sfreq)
            if args.eeg_mode == "patched":
                eeg = maybe_patch_eeg(eeg, seq_len=args.eeg_seq_len, patch_len=args.eeg_patch_len)

            fmri = extract_roi_timeseries(
                fmri_nii_path=fmri_nii,
                labels_img=labels_img,
                tr=args.tr,
                discard_initial_trs=args.discard_initial_trs,
                standardize_fmri=args.standardize_fmri,
                fmri_target_t=args.fmri_target_t,
            )
            fmri = resample_fmri_if_needed(fmri, args.fmri_target_rois, None)

            sample_id = f"{subject}_{task}"
            eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
            fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
            np.save(eeg_out_path, eeg)
            np.save(fmri_out_path, fmri)

            records.append(
                build_sample_record(
                    sample_id=sample_id,
                    subject=subject,
                    task=task,
                    trial_type=task,
                    eeg_rel_path=eeg_out_path.relative_to(out_root),
                    fmri_rel_path=fmri_out_path.relative_to(out_root),
                    label=TASK_LABELS[task],
                    label_name=task,
                    eeg=eeg,
                    fmri=fmri,
                )
            )
            continue

        events = load_task_events(ds_root, task)
        eeg = crop_eeg_to_task(eeg, task, sfreq=sfreq)
        fmri_full = extract_roi_timeseries(
            fmri_nii_path=fmri_nii,
            labels_img=labels_img,
            tr=args.tr,
            discard_initial_trs=0,
            standardize_fmri=args.standardize_fmri,
            fmri_target_t=None,
        )

        for block_idx, row in events.reset_index(drop=True).iterrows():
            onset_sec = float(row["onset"]) - args.protocol_offset_sec
            duration_sec = float(row["duration"])
            trial_type = str(row["trial_type"]).strip()
            eeg_block = slice_eeg_block(eeg, sfreq=sfreq, start_sec=onset_sec, duration_sec=duration_sec)
            if args.eeg_mode == "patched":
                eeg_block = maybe_patch_eeg(eeg_block, seq_len=args.eeg_seq_len, patch_len=args.eeg_patch_len)

            fmri_block = slice_fmri_block(fmri_full, tr=args.tr, start_sec=onset_sec, duration_sec=duration_sec)
            fmri_block = resample_fmri_if_needed(fmri_block, args.fmri_target_rois, args.fmri_target_t)

            if args.label_mode == "binary_rest_task":
                label, label_name = resolve_binary_label(trial_type)
            else:
                label = TASK_LABELS[task]
                label_name = task

            sample_id = f"{subject}_{task}_block-{block_idx:02d}"
            eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
            fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
            np.save(eeg_out_path, eeg_block)
            np.save(fmri_out_path, fmri_block)

            records.append(
                build_sample_record(
                    sample_id=sample_id,
                    subject=subject,
                    task=task,
                    trial_type=trial_type,
                    eeg_rel_path=eeg_out_path.relative_to(out_root),
                    fmri_rel_path=fmri_out_path.relative_to(out_root),
                    label=label,
                    label_name=label_name,
                    eeg=eeg_block,
                    fmri=fmri_block,
                )
            )

    if not records:
        raise RuntimeError("No samples were exported. Check subject IDs, task names, and input paths.")

    manifest = pd.DataFrame(record.__dict__ for record in records)
    manifest.to_csv(out_root / "manifest_all.csv", index=False)


if __name__ == "__main__":
    main()
