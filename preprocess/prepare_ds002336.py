from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from preprocess_common import (
    add_atlas_args,
    add_common_fmri_args,
    add_dataset_io_args,
    add_eeg_patch_args,
    add_fmri_roi_resample_args,
    add_subject_args,
    add_subject_packing_and_split_args,
    add_training_ready_arg,
    extract_roi_timeseries,
    find_subjects,
    get_atlas_labels_img,
    load_bold_volume,
    preprocess_fmri_volume,
    prepare_training_ready_eeg,
    prepare_training_ready_fmri,
    resample_fmri_if_needed,
    stack_subject_samples,
    write_subject_memmap_pack,
    write_loso_splits,
    write_subject_splits,
)


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
    training_ready: bool = False


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    subject_path: str
    sample_count: int
    eeg_shape: str
    fmri_shape: str
    label_shape: str
    training_ready: bool = False


@dataclass(frozen=True)
class SkippedBlockRecord:
    subject: str
    task: str
    block_index: int
    trial_type: str
    onset_sec: float
    duration_sec: float
    reason: str


@dataclass(frozen=True)
class MissingPairRecord:
    subject: str
    task: str
    eeg_path: str
    fmri_path: str
    reason: str


@dataclass(frozen=True)
class WindowPlacement:
    eeg_start_sec: float
    fmri_start_sec: float
    duration_sec: float
    protocol_onset_sec: float
    shift_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds002336 EEG/fMRI pairs for this repository.")
    add_dataset_io_args(parser, ds_root_help="Path to ds002336 root.", output_root_help="Output directory for converted arrays and manifests.")
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
    add_subject_args(parser, subject_example="sub-xp101")
    add_atlas_args(parser)
    parser.add_argument(
        "--fmri-source",
        default="raw",
        choices=["raw", "spm_unsmoothed", "spm_smoothed"],
        help="raw reads original func/*_bold.nii.gz; spm_* reads derivatives/spm12_preproc outputs.",
    )
    parser.add_argument(
        "--fmri-preproc-root",
        type=Path,
        default=None,
        help="Optional root for SPM preprocessed fMRI. Defaults to <ds-root>/derivatives/spm12_preproc when --fmri-source is not raw.",
    )
    add_common_fmri_args(
        parser,
        default_fmri_mode="roi",
        tr_help="fMRI repetition time.",
        standardize_help="Apply standardization inside the ROI masker.",
        fmri_max_shape_help="Maximum center-cropped spatial shape after volume resampling.",
    )
    parser.add_argument(
        "--discard-initial-trs",
        type=int,
        default=1,
        help="Initial BOLD volumes to discard before any slicing. Defaults to 1 because ds002336 README states the scanner starts 2 seconds before protocol onset and TR=2s.",
    )
    parser.add_argument(
        "--protocol-offset-sec",
        type=float,
        default=2.0,
        help="Seconds to subtract from task events TSV onsets to map them onto protocol time. Defaults to 2.0 for ds002336.",
    )
    add_fmri_roi_resample_args(parser)
    add_eeg_patch_args(
        parser,
        default_eeg_mode="continuous",
        default_seq_len=None,
        default_patch_len=None,
        seq_len_help="Optional EEG patch count when eeg-mode=patched. Defaults to 30 for run mode and to block duration in seconds for block mode.",
        patch_len_help="Optional EEG patch length when eeg-mode=patched. Defaults to the EEG sampling rate so each patch spans about one second.",
    )
    parser.add_argument(
        "--drop-ecg",
        action="store_true",
        help="Drop ECG and other non-EEG channels when reading the preprocessed BrainVision files.",
    )
    add_subject_packing_and_split_args(
        parser,
        pack_help="Pack all exported samples of the same subject into one directory of memmap-friendly NPY files, so downstream loading is subject-packed.",
        split_help="Optional split generation after preprocessing.",
        train_subjects=7,
        val_subjects=2,
        test_subjects=1,
    )
    add_training_ready_arg(parser)
    return parser.parse_args()


def resolve_fmri_path(ds_root: Path, subject: str, task: str, args: argparse.Namespace) -> Path:
    if args.fmri_source == "raw":
        return ds_root / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"

    fmri_preproc_root = args.fmri_preproc_root.resolve() if args.fmri_preproc_root is not None else (ds_root / "derivatives" / "spm12_preproc")
    subject_dir = fmri_preproc_root / subject
    if args.fmri_source == "spm_smoothed":
        flat_final = subject_dir / f"{subject}_task-{task}_bold.nii"
        if flat_final.exists():
            return flat_final
        legacy_task_dir = subject_dir / f"task-{task}"
        legacy_final = legacy_task_dir / "fmri_final.nii"
        if legacy_final.exists():
            return legacy_final
        return legacy_task_dir / f"swratrim_{subject}_task-{task}_bold.nii"

    legacy_task_dir = subject_dir / f"task-{task}"
    legacy_unsmoothed = legacy_task_dir / f"wratrim_{subject}_task-{task}_bold.nii"
    return legacy_unsmoothed


def _normalize_marker_description(description: str) -> str:
    return str(description).strip().upper().replace(" ", "")


def detect_eeg_protocol_start_sec(raw: mne.io.BaseRaw) -> float:
    first_s99: float | None = None
    first_s2: float | None = None
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        marker = _normalize_marker_description(description)
        if marker.endswith("/S99") and first_s99 is None:
            first_s99 = float(onset)
        if marker.endswith("/S2") and first_s2 is None:
            first_s2 = float(onset)
    if first_s99 is not None:
        return first_s99
    if first_s2 is not None:
        return max(0.0, first_s2 - 20.0)
    raise ValueError("Could not locate EEG protocol start marker S99 or fallback S2 in BrainVision annotations")


def load_eeg(eeg_vhdr_path: Path, drop_ecg: bool) -> tuple[np.ndarray, float, float]:
    raw = mne.io.read_raw_brainvision(str(eeg_vhdr_path), preload=True, verbose="ERROR")
    protocol_start_sec = detect_eeg_protocol_start_sec(raw)
    if drop_ecg:
        raw = raw.pick("eeg")
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"]), float(protocol_start_sec)


def crop_eeg_to_task(data: np.ndarray, task: str, sfreq: float) -> np.ndarray:
    duration_sec = TASK_DURATIONS_SEC[task]
    target_samples = int(round(duration_sec * sfreq))
    if data.shape[1] < target_samples:
        raise ValueError(f"EEG samples shorter than expected for task {task}: {data.shape[1]} < {target_samples}")
    return data[:, :target_samples]


def crop_eeg_to_duration(data: np.ndarray, sfreq: float, duration_sec: float) -> np.ndarray:
    target_samples = int(round(duration_sec * sfreq))
    if data.shape[1] < target_samples:
        raise ValueError(f"EEG samples shorter than requested duration: {data.shape[1]} < {target_samples}")
    return data[:, :target_samples]


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    if data.shape[1] != target_len:
        data = resample(data, target_len, axis=1)
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def resolve_eeg_patch_params(
    sfreq: float,
    requested_seq_len: int | None,
    requested_patch_len: int | None,
    sample_mode: str,
    duration_sec: float | None = None,
) -> tuple[int, int]:
    patch_len = requested_patch_len if requested_patch_len is not None else int(round(sfreq))
    if patch_len <= 0:
        raise ValueError(f"EEG patch length must be positive, got {patch_len}")

    if requested_seq_len is not None:
        seq_len = requested_seq_len
    elif sample_mode == "block":
        if duration_sec is None:
            raise ValueError("Block-mode EEG patch inference requires duration_sec.")
        seq_len = max(1, int(round(duration_sec)))
    else:
        seq_len = 30

    if seq_len <= 0:
        raise ValueError(f"EEG sequence length must be positive, got {seq_len}")
    return seq_len, patch_len


def load_task_events(ds_root: Path, task: str) -> pd.DataFrame:
    events_path = ds_root / f"task-{task}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(f"Task events TSV not found: {events_path}")
    rows: list[dict[str, object]] = []
    with open(events_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Events TSV is empty: {events_path}")
        index_by_name = {name: idx for idx, name in enumerate(header)}
        required_columns = ["onset", "duration", "trial_type"]
        missing = [name for name in required_columns if name not in index_by_name]
        if missing:
            raise ValueError(f"Events TSV missing required columns {missing}: {events_path}")
        for parts in reader:
            if not parts or not any(cell.strip() for cell in parts):
                continue
            onset = parts[index_by_name["onset"]].strip() if index_by_name["onset"] < len(parts) else ""
            duration = parts[index_by_name["duration"]].strip() if index_by_name["duration"] < len(parts) else ""
            trial_type = parts[index_by_name["trial_type"]].strip() if index_by_name["trial_type"] < len(parts) else ""
            if not onset or not duration or not trial_type:
                continue
            rows.append(
                {
                    "onset": float(onset),
                    "duration": float(duration),
                    "trial_type": trial_type,
                }
            )
    if not rows:
        raise ValueError(f"No valid event rows found in {events_path}")
    return pd.DataFrame(rows)


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


def slice_fmri_volume_block(volume: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    if start < 0 or end > volume.shape[3]:
        raise ValueError(f"fMRI volume block slice out of range: start={start}, end={end}, total={volume.shape[3]}")
    return volume[:, :, :, start:end].astype(np.float32)


def compute_shifted_window(
    eeg_total_sec: float,
    eeg_protocol_start_sec: float,
    fmri_total_sec: float,
    protocol_onset_sec: float,
    duration_sec: float,
) -> WindowPlacement | None:
    eeg_start_sec = eeg_protocol_start_sec + protocol_onset_sec
    fmri_start_sec = protocol_onset_sec

    if eeg_start_sec < 0 or fmri_start_sec < 0:
        return None

    eeg_overflow = max(0.0, eeg_start_sec + duration_sec - eeg_total_sec)
    fmri_overflow = max(0.0, fmri_start_sec + duration_sec - fmri_total_sec)
    shift_sec = max(eeg_overflow, fmri_overflow)

    if shift_sec > 0:
        eeg_start_sec -= shift_sec
        fmri_start_sec -= shift_sec
        protocol_onset_sec -= shift_sec

    if eeg_start_sec < 0 or fmri_start_sec < 0:
        return None
    if eeg_start_sec + duration_sec > eeg_total_sec + 1e-6:
        return None
    if fmri_start_sec + duration_sec > fmri_total_sec + 1e-6:
        return None

    return WindowPlacement(
        eeg_start_sec=float(eeg_start_sec),
        fmri_start_sec=float(fmri_start_sec),
        duration_sec=float(duration_sec),
        protocol_onset_sec=float(protocol_onset_sec),
        shift_sec=float(shift_sec),
    )


def block_fits_eeg(data: np.ndarray, sfreq: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    end = start + length
    return start >= 0 and end <= data.shape[1]


def block_fits_fmri_matrix(series: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    return start >= 0 and end <= series.shape[1]


def block_fits_fmri_volume(volume: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    return start >= 0 and end <= volume.shape[3]


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
    training_ready: bool,
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
        training_ready=bool(training_ready),
    )


def iter_subject_task_pairs(subjects: Iterable[str], tasks: Iterable[str]) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        for task in tasks:
            yield subject, task


def main() -> None:
    args = parse_args()
    if args.label_mode == "binary_rest_task" and args.sample_mode != "block":
        raise ValueError("binary_rest_task requires --sample-mode block because each run contains both rest and task blocks.")
    if args.fmri_mode == "volume" and args.fmri_target_rois is not None:
        raise ValueError("--fmri-target-rois is only valid when --fmri-mode=roi")

    ds_root = args.ds_root.resolve()
    out_root = args.output_root.resolve()
    eeg_out_dir = out_root / "eeg"
    fmri_out_dir = out_root / "fmri"
    packed_out_dir = out_root / "subjects"
    atlas_cache_dir = out_root / "atlas_cache"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.pack_subject_files:
        packed_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        eeg_out_dir.mkdir(parents=True, exist_ok=True)
        fmri_out_dir.mkdir(parents=True, exist_ok=True)

    labels_img = get_atlas_labels_img(args.atlas_labels_img, atlas_cache_dir, args.n_rois) if args.fmri_mode == "roi" else ""
    subjects = find_subjects(ds_root, args.subjects)
    fmri_discard_initial_trs = int(args.discard_initial_trs)
    protocol_offset_sec = float(args.protocol_offset_sec)
    if args.fmri_source != "raw":
        if fmri_discard_initial_trs != 0 or abs(protocol_offset_sec) > 1e-6:
            print(
                "Using SPM-preprocessed fMRI: overriding discard_initial_trs to 0 and protocol_offset_sec to 0.0 "
                "because the first lead-in TR has already been removed in the SPM pipeline."
            )
        fmri_discard_initial_trs = 0
        protocol_offset_sec = 0.0
        if args.fmri_mode == "volume" and tuple(args.fmri_max_shape) != (79, 95, 79):
            print(
                f"Warning: SPM-preprocessed MNI volumes are typically about 79x95x79, but fmri_max_shape={tuple(args.fmri_max_shape)}. "
                "Further cropping may remove brain coverage before training."
            )

    records: list[SampleRecord] = []
    subject_records: list[SubjectRecord] = []
    skipped_blocks: list[SkippedBlockRecord] = []
    missing_pairs: list[MissingPairRecord] = []
    for subject in tqdm(subjects, desc="Preparing ds002336"):
        packed_eeg_samples: list[np.ndarray] = []
        packed_fmri_samples: list[np.ndarray] = []
        packed_labels: list[int] = []
        packed_sample_ids: list[str] = []
        packed_tasks: list[str] = []
        packed_trial_types: list[str] = []

        for task in args.tasks:
            eeg_vhdr = ds_root / "derivatives" / subject / "eeg_pp" / f"{subject}_task-{task}_eeg_pp.vhdr"
            fmri_nii = resolve_fmri_path(ds_root, subject, task, args)
            missing_reasons: list[str] = []
            if not eeg_vhdr.exists():
                missing_reasons.append("missing_eeg")
            if not fmri_nii.exists():
                missing_reasons.append("missing_fmri")
            if missing_reasons:
                missing_pairs.append(
                    MissingPairRecord(
                        subject=subject,
                        task=task,
                        eeg_path=str(eeg_vhdr),
                        fmri_path=str(fmri_nii),
                        reason="+".join(missing_reasons),
                    )
                )
                continue

            eeg, sfreq, eeg_protocol_start_sec = load_eeg(eeg_vhdr, drop_ecg=args.drop_ecg)
            if args.sample_mode == "run":
                eeg = crop_eeg_to_task(eeg, task, sfreq=sfreq)
                if args.eeg_mode == "patched":
                    eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(
                        sfreq=sfreq,
                        requested_seq_len=args.eeg_seq_len,
                        requested_patch_len=args.eeg_patch_len,
                        sample_mode=args.sample_mode,
                    )
                    eeg = maybe_patch_eeg(eeg, seq_len=eeg_seq_len, patch_len=eeg_patch_len)
                eeg = prepare_training_ready_eeg(eeg, enabled=bool(args.training_ready))

                if args.fmri_mode == "roi":
                    fmri = extract_roi_timeseries(
                        fmri_nii_path=fmri_nii,
                        labels_img=labels_img,
                        tr=args.tr,
                        standardize_fmri=args.standardize_fmri,
                        discard_initial_trs=fmri_discard_initial_trs,
                        fmri_target_t=args.fmri_target_t,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                    fmri = resample_fmri_if_needed(
                        fmri,
                        args.fmri_target_rois,
                        None,
                        allow_roi_resample=args.allow_fmri_roi_resample,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                else:
                    fmri_volume, voxel_size = load_bold_volume(
                        fmri_nii_path=fmri_nii,
                        discard_initial_trs=fmri_discard_initial_trs,
                    )
                    fmri = preprocess_fmri_volume(
                        fmri_volume,
                        voxel_size=voxel_size,
                        source_tr=float(voxel_size[3]) if len(voxel_size) > 3 else float(args.tr),
                        target_voxel_size=tuple(args.fmri_voxel_size),
                        target_tr=float(args.tr),
                        max_shape=tuple(args.fmri_max_shape),
                        use_float16=bool(args.fmri_float16),
                    )
                fmri = prepare_training_ready_fmri(fmri, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))

                sample_id = f"{subject}_{task}"
                if args.pack_subject_files:
                    packed_eeg_samples.append(eeg.astype(np.float32))
                    packed_fmri_samples.append(fmri.astype(np.float32))
                    packed_labels.append(int(TASK_LABELS[task]))
                    packed_sample_ids.append(sample_id)
                    packed_tasks.append(task)
                    packed_trial_types.append(task)
                else:
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
                            training_ready=bool(args.training_ready),
                        )
                    )
                continue

            events = load_task_events(ds_root, task)
            if args.fmri_mode == "roi":
                fmri_full = extract_roi_timeseries(
                    fmri_nii_path=fmri_nii,
                    labels_img=labels_img,
                    tr=args.tr,
                    standardize_fmri=args.standardize_fmri,
                    discard_initial_trs=fmri_discard_initial_trs,
                    fmri_target_t=None,
                    allow_time_resample=args.allow_fmri_time_resample,
                )
            else:
                fmri_volume, voxel_size = load_bold_volume(
                    fmri_nii_path=fmri_nii,
                    discard_initial_trs=fmri_discard_initial_trs,
                )
                fmri_full = preprocess_fmri_volume(
                    fmri_volume,
                    voxel_size=voxel_size,
                    source_tr=float(voxel_size[3]) if len(voxel_size) > 3 else float(args.tr),
                    target_voxel_size=tuple(args.fmri_voxel_size),
                    target_tr=float(args.tr),
                    max_shape=tuple(args.fmri_max_shape),
                    use_float16=bool(args.fmri_float16),
                )

            eeg_total_sec = float(eeg.shape[1]) / float(sfreq)
            fmri_total_sec = float(fmri_full.shape[1]) * float(args.tr) if args.fmri_mode == "roi" else float(fmri_full.shape[3]) * float(args.tr)

            for block_idx, row in events.reset_index(drop=True).iterrows():
                onset_sec = float(row["onset"]) - protocol_offset_sec
                duration_sec = float(row["duration"])
                trial_type = str(row["trial_type"]).strip()
                placement = compute_shifted_window(
                    eeg_total_sec=eeg_total_sec,
                    eeg_protocol_start_sec=eeg_protocol_start_sec,
                    fmri_total_sec=fmri_total_sec,
                    protocol_onset_sec=onset_sec,
                    duration_sec=duration_sec,
                )
                if placement is None:
                    skipped_blocks.append(
                        SkippedBlockRecord(
                            subject=subject,
                            task=task,
                            block_index=int(block_idx),
                            trial_type=trial_type,
                            onset_sec=float(onset_sec),
                            duration_sec=float(duration_sec),
                            reason="paired_window_out_of_range",
                        )
                    )
                    continue
                eeg_block = slice_eeg_block(eeg, sfreq=sfreq, start_sec=placement.eeg_start_sec, duration_sec=duration_sec)
                if args.eeg_mode == "patched":
                    eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(
                        sfreq=sfreq,
                        requested_seq_len=args.eeg_seq_len,
                        requested_patch_len=args.eeg_patch_len,
                        sample_mode=args.sample_mode,
                        duration_sec=duration_sec,
                    )
                    eeg_block = maybe_patch_eeg(eeg_block, seq_len=eeg_seq_len, patch_len=eeg_patch_len)
                eeg_block = prepare_training_ready_eeg(eeg_block, enabled=bool(args.training_ready))

                if args.fmri_mode == "roi":
                    fmri_block = slice_fmri_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=duration_sec)
                    fmri_block = resample_fmri_if_needed(
                        fmri_block,
                        args.fmri_target_rois,
                        args.fmri_target_t,
                        allow_roi_resample=args.allow_fmri_roi_resample,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                else:
                    fmri_block = slice_fmri_volume_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=duration_sec)
                fmri_block = prepare_training_ready_fmri(fmri_block, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))

                if args.label_mode == "binary_rest_task":
                    label, label_name = resolve_binary_label(trial_type)
                else:
                    label = TASK_LABELS[task]
                    label_name = task

                sample_id = f"{subject}_{task}_block-{block_idx:02d}"
                if args.pack_subject_files:
                    packed_eeg_samples.append(eeg_block.astype(np.float32))
                    packed_fmri_samples.append(fmri_block.astype(np.float32))
                    packed_labels.append(int(label))
                    packed_sample_ids.append(sample_id)
                    packed_tasks.append(task)
                    packed_trial_types.append(trial_type)
                else:
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
                            training_ready=bool(args.training_ready),
                        )
                    )

        if args.pack_subject_files and packed_eeg_samples:
            packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
            packed_fmri = stack_subject_samples(packed_fmri_samples, name="fMRI")
            packed_labels_array = np.asarray(packed_labels, dtype=np.int64)
            packed_sample_ids_array = np.asarray(packed_sample_ids)
            packed_tasks_array = np.asarray(packed_tasks)
            packed_trial_types_array = np.asarray(packed_trial_types)

            subject_path = write_subject_memmap_pack(
                packed_out_dir / subject,
                {
                    "eeg": packed_eeg,
                    "fmri": packed_fmri,
                    "labels": packed_labels_array,
                    "sample_id": packed_sample_ids_array,
                    "task": packed_tasks_array,
                    "trial_type": packed_trial_types_array,
                },
            )
            subject_records.append(
                SubjectRecord(
                    subject=subject,
                    subject_path=subject_path.relative_to(out_root).as_posix(),
                    sample_count=int(packed_labels_array.shape[0]),
                    eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
                    fmri_shape="x".join(str(dim) for dim in packed_fmri.shape),
                    label_shape="x".join(str(dim) for dim in packed_labels_array.shape),
                    training_ready=bool(args.training_ready),
                )
            )

    if not records and not subject_records:
        raise RuntimeError("No samples were exported. Check subject IDs, task names, and input paths.")

    if subject_records:
        pd.DataFrame(record.__dict__ for record in subject_records).to_csv(out_root / "manifest_all.csv", index=False)
    else:
        pd.DataFrame(record.__dict__ for record in records).to_csv(out_root / "manifest_all.csv", index=False)
    if skipped_blocks:
        pd.DataFrame(record.__dict__ for record in skipped_blocks).to_csv(out_root / "skipped_blocks.csv", index=False)
    if missing_pairs:
        pd.DataFrame(record.__dict__ for record in missing_pairs).to_csv(out_root / "missing_pairs.csv", index=False)
        print(f"Skipped {len(missing_pairs)} subject-task pairs because EEG/fMRI files were not both present.")

    if args.split_mode == "subject":
        split_dir = args.split_output_dir.resolve() if args.split_output_dir else (out_root / "splits_subjectwise")
        write_subject_splits(
            manifest_path=out_root / "manifest_all.csv",
            output_dir=split_dir,
            train_subjects=int(args.train_subjects),
            val_subjects=int(args.val_subjects),
            test_subjects=int(args.test_subjects),
        )
    elif args.split_mode == "loso":
        split_dir = args.split_output_dir.resolve() if args.split_output_dir else (out_root / "loso_subjectwise")
        write_loso_splits(
            manifest_path=out_root / "manifest_all.csv",
            output_dir=split_dir,
            val_subjects=int(args.val_subjects),
        )


if __name__ == "__main__":
    main()
