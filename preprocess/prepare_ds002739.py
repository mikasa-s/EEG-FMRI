from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm

from preprocess_common import (
    add_atlas_args,
    add_common_fmri_args,
    add_dataset_io_args,
    add_eeg_patch_args,
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
    stack_subject_samples,
    write_subject_memmap_pack,
    write_loso_splits,
    write_subject_splits,
)


DOT_DIRECTION_LABELS = {
    0: (0, "direction_0deg"),
    180: (1, "direction_180deg"),
    1: (0, "left"),
    2: (1, "right"),
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
    window_sec: float
    trial_index: int
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
class RunSummary:
    subject: str
    run: str
    bold_shape: str
    voxel_size: str
    target_voxel_size: str
    exported_fmri_shape: str
    eeg_shape: str
    eeg_fmri_offset_sec: float
    event_counts: str
    eeg_sfreq_hz: float
    fmri_target_tr_sec: float
    valid_trial_count: int
    fmri_trial_count: int


@dataclass(frozen=True)
class SubjectPreparationResult:
    subject: str
    records: list[SampleRecord]
    subject_record: SubjectRecord | None
    summaries: list[RunSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds002739 EEG/fMRI pairs with normalized NeuroSTORM-style volume preprocessing.")
    add_dataset_io_args(parser, ds_root_help="Path to ds002739 root.", output_root_help="Output directory for exported arrays and manifests.")
    add_subject_args(parser, subject_example="sub-01")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional run IDs such as run-01 run-02. Defaults to all runs found per subject.",
    )
    parser.add_argument(
        "--fmri-event-type",
        default="dot_stim_validtrials",
        help="fMRI auxiliary event used to pair each EEG trial. For ds002739 dotdirection labels this should stay at dot_stim_validtrials.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=8.0,
        help="Maximum window length in seconds. Actual windows are truncated to avoid crossing the next tracked event boundary.",
    )
    parser.add_argument(
        "--eeg-window-sec",
        type=float,
        default=None,
        help="Optional fixed EEG window length in seconds. Defaults to --window-sec.",
    )
    parser.add_argument(
        "--fmri-window-sec",
        type=float,
        default=None,
        help="Optional fixed fMRI window length in seconds. Defaults to --window-sec.",
    )
    parser.add_argument(
        "--window-margin-sec",
        type=float,
        default=0.0,
        help="Optional safety margin removed from the next-event boundary when computing the usable window.",
    )
    parser.add_argument(
        "--min-window-sec",
        type=float,
        default=1.0,
        help="Minimum usable window length. Shorter windows are skipped.",
    )
    add_atlas_args(parser)
    add_common_fmri_args(
        parser,
        default_fmri_mode="volume",
        tr_help="Target fMRI repetition time in seconds after preprocessing.",
        standardize_help="Apply standardization inside the ROI masker when fmri-mode=roi.",
        fmri_max_shape_help="Maximum center-cropped spatial shape after resampling. Dimensions smaller than this are not padded during preprocessing.",
    )
    add_eeg_patch_args(
        parser,
        default_eeg_mode="patched",
        default_seq_len=20,
        default_patch_len=200,
        seq_len_help="EEG patch count when eeg-mode=patched.",
        patch_len_help="EEG patch length when eeg-mode=patched.",
    )
    parser.add_argument("--eeg-target-sfreq", type=float, default=200.0, help="Target EEG sampling rate in Hz.")
    parser.add_argument("--eeg-lfreq", type=float, default=0.5, help="EEG band-pass low cutoff in Hz.")
    parser.add_argument("--eeg-hfreq", type=float, default=40.0, help="EEG band-pass high cutoff in Hz.")
    add_subject_packing_and_split_args(
        parser,
        pack_help="Pack all runs of the same subject into one directory of memmap-friendly NPY files with arrays stacked along axis 0.",
        split_help="Optional split generation after preprocessing. subject writes train/val/test manifests; loso writes per-fold manifests.",
        train_subjects=21,
        val_subjects=2,
        test_subjects=1,
    )
    add_training_ready_arg(parser)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of subject-level worker processes. Use 1 to keep processing serial.")
    return parser.parse_args()


def load_mat_payload(path: Path) -> dict:
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def load_electrode_template(ds_root: Path) -> list[str]:
    electrode_path = ds_root / "additional_files" / "electrode_info.elp"
    if not electrode_path.exists():
        raise FileNotFoundError(f"Electrode template not found: {electrode_path}")
    electrode_names: list[str] = []
    with open(electrode_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            electrode_names.append(stripped.split()[0])
    if not electrode_names:
        raise ValueError(f"No electrode names found in {electrode_path}")
    return electrode_names


def normalize_excluded_channels(raw_value: object) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return [] if int(raw_value) == 0 else [int(raw_value)]
    if isinstance(raw_value, np.ndarray):
        values = raw_value.reshape(-1).tolist()
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        return []

    excluded: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            excluded.extend(normalize_excluded_channels(value))
            continue
        channel_index = int(value)
        if channel_index != 0:
            excluded.append(channel_index)
    return sorted(set(excluded))


def compute_common_electrodes(ds_root: Path, subjects: list[str], requested_runs: list[str] | None, electrode_template: list[str]) -> list[str]:
    common = set(electrode_template)
    file_count = 0
    for subject, run in iter_subject_runs(subjects, ds_root, requested_runs):
        eeg_path = ds_root / subject / "EEG" / f"EEG_data_{subject}_{run}.mat"
        if not eeg_path.exists():
            continue
        payload = loadmat(str(eeg_path), variable_names=["excludedchannels"], squeeze_me=True, struct_as_record=False)
        excluded = normalize_excluded_channels(payload.get("excludedchannels", []))
        keep = {name for index, name in enumerate(electrode_template, start=1) if index not in set(excluded)}
        common &= keep
        file_count += 1

    if file_count == 0:
        raise RuntimeError("No EEG files found while computing shared electrode intersection.")
    ordered_common = [name for name in electrode_template if name in common]
    if not ordered_common:
        raise RuntimeError("No shared EEG electrodes remain after applying excludedchannels across the selected files.")
    return ordered_common


def save_common_electrode_manifest(out_root: Path, electrode_template: list[str], common_electrodes: list[str]) -> None:
    rows = [
        {
            "template_index_1based": index,
            "channel_name": name,
            "kept": name in set(common_electrodes),
        }
        for index, name in enumerate(electrode_template, start=1)
    ]
    pd.DataFrame(rows).to_csv(out_root / "eeg_channel_intersection.csv", index=False)


def load_eeg_data(eeg_mat_path: Path, electrode_template: list[str], common_electrodes: list[str]) -> tuple[np.ndarray, float]:
    payload = load_mat_payload(eeg_mat_path)
    eeg_struct = payload["EEGdata"]
    eeg = np.asarray(eeg_struct.Y, dtype=np.float32)
    sfreq = float(payload.get("fs", getattr(eeg_struct, "fs", 1000.0)))
    if eeg.ndim != 2:
        raise ValueError(f"Expected EEG data with shape [channels, samples], got {eeg.shape} from {eeg_mat_path}")
    excluded = normalize_excluded_channels(payload.get("excludedchannels", []))
    kept_electrodes = [name for index, name in enumerate(electrode_template, start=1) if index not in set(excluded)]
    if eeg.shape[0] != len(kept_electrodes):
        raise ValueError(
            f"EEG channel count mismatch for {eeg_mat_path}: data has {eeg.shape[0]} channels, but template minus excludedchannels yields {len(kept_electrodes)}"
        )
    kept_index_by_name = {name: index for index, name in enumerate(kept_electrodes)}
    missing = [name for name in common_electrodes if name not in kept_index_by_name]
    if missing:
        raise ValueError(f"Shared EEG electrodes missing in {eeg_mat_path}: {missing}")
    selected_indices = [kept_index_by_name[name] for name in common_electrodes]
    return eeg[selected_indices].astype(np.float32), sfreq


def load_eeg_events(events_mat_path: Path) -> dict[str, np.ndarray]:
    payload = load_mat_payload(events_mat_path)
    return {key: np.asarray(value).reshape(-1) for key, value in payload.items() if not key.startswith("__")}


def build_eeg_trial_table(eeg_events: dict[str, np.ndarray]) -> pd.DataFrame:
    required = ["tstim", "dotdirection"]
    missing = [key for key in required if key not in eeg_events]
    if missing:
        raise ValueError(f"EEG events file is missing required fields for EEG-led labeling: {missing}")

    tstim = np.asarray(eeg_events["tstim"], dtype=np.float64).reshape(-1)
    dotdirection = np.asarray(eeg_events["dotdirection"], dtype=np.float64).reshape(-1)
    trial_count = min(len(tstim), len(dotdirection))
    rows: list[dict[str, float | int | str]] = []

    for trial_index in range(trial_count):
        onset_ms = tstim[trial_index]
        direction = dotdirection[trial_index]
        if np.isnan(onset_ms) or np.isnan(direction):
            continue
        direction_value = int(direction)
        if direction_value not in DOT_DIRECTION_LABELS:
            continue
        label, label_name = DOT_DIRECTION_LABELS[direction_value]
        rows.append(
            {
                "trial_index": trial_index,
                "eeg_onset_sec": float(onset_ms) / 1000.0,
                "dotdirection": direction_value,
                "label": label,
                "label_name": label_name,
            }
        )

    if not rows:
        raise ValueError("No valid EEG trials were found from tstim and dotdirection.")

    return pd.DataFrame(rows).sort_values("eeg_onset_sec").reset_index(drop=True)


def bandpass_filter_eeg(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    nyquist = sfreq / 2.0
    if not (0.0 < l_freq < h_freq < nyquist):
        raise ValueError(f"Invalid EEG band-pass range [{l_freq}, {h_freq}] for sfreq={sfreq}")
    sos = butter(N=4, Wn=[l_freq, h_freq], btype="bandpass", output="sos", fs=sfreq)
    return sosfiltfilt(sos, data, axis=1).astype(np.float32)


def resample_eeg(data: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if source_sfreq <= 0 or target_sfreq <= 0:
        raise ValueError(f"EEG sampling rates must be positive, got source={source_sfreq}, target={target_sfreq}")
    if abs(source_sfreq - target_sfreq) < 1e-6:
        return data.astype(np.float32)
    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(1000)
    return resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=1).astype(np.float32)


def preprocess_eeg(data: np.ndarray, source_sfreq: float, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    filtered = bandpass_filter_eeg(data, sfreq=source_sfreq, l_freq=args.eeg_lfreq, h_freq=args.eeg_hfreq)
    downsampled = resample_eeg(filtered, source_sfreq=source_sfreq, target_sfreq=args.eeg_target_sfreq)
    return downsampled.astype(np.float32), float(args.eeg_target_sfreq)


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


def load_fmri_events(events_tsv_path: Path, event_type: str) -> pd.DataFrame:
    events = pd.read_csv(events_tsv_path, sep="\t")
    required = {"onset", "duration", "trial_type"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing required columns {required} in {events_tsv_path}")
    events = events.dropna(subset=["onset", "duration", "trial_type"]).copy()
    filtered = events[events["trial_type"] == event_type].copy()
    filtered["onset"] = filtered["onset"].astype(np.float64)
    filtered["duration"] = filtered["duration"].astype(np.float64)
    return filtered.sort_values("onset").reset_index(drop=True)


def estimate_eeg_fmri_offset_sec(eeg_trials: pd.DataFrame, fmri_events: pd.DataFrame) -> float:
    eeg_tstim = eeg_trials["eeg_onset_sec"].to_numpy(dtype=np.float64)
    fmri_tstim = fmri_events["onset"].to_numpy(dtype=np.float64)
    if len(eeg_tstim) == 0 or len(fmri_tstim) == 0:
        raise ValueError("Cannot estimate alignment offset because EEG or fMRI stimulus events are missing.")
    count = min(len(eeg_tstim), len(fmri_tstim))
    return float(np.median(eeg_tstim[:count] - fmri_tstim[:count]))


def compute_event_window_sec(onsets_sec: np.ndarray, event_index: int, max_window_sec: float, margin_sec: float, min_window_sec: float) -> float | None:
    onset = float(onsets_sec[event_index])
    usable = float(max_window_sec)
    if event_index + 1 < len(onsets_sec):
        next_onset = float(onsets_sec[event_index + 1])
        usable = min(usable, max(0.0, next_onset - onset - margin_sec))
    if usable < min_window_sec:
        return None
    return usable


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


def slice_fmri_volume_window(volume: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = max(1, int(round(duration_sec / tr)))
    end = start + length
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {volume.shape}")
    if start < 0 or end > volume.shape[3]:
        raise ValueError(f"fMRI volume slice out of range: start={start}, end={end}, total={volume.shape[3]}")
    return volume[:, :, :, start:end]


def pad_or_crop_eeg(data: np.ndarray, target_len: int) -> np.ndarray:
    current_len = int(data.shape[1])
    if current_len == target_len:
        return data.astype(np.float32)
    if current_len > target_len:
        return data[:, :target_len].astype(np.float32)
    padded = np.zeros((data.shape[0], target_len), dtype=np.float32)
    padded[:, :current_len] = data.astype(np.float32)
    return padded


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    data = pad_or_crop_eeg(data, target_len=target_len)
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def resolve_eeg_patch_params(args: argparse.Namespace) -> tuple[int, int]:
    seq_len = int(args.eeg_seq_len)
    patch_len = int(args.eeg_patch_len)
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
    window_sec: float,
    trial_index: int,
    training_ready: bool,
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
        window_sec=float(window_sec),
        trial_index=int(trial_index),
        training_ready=bool(training_ready),
    )


def iter_subject_runs(subjects: Iterable[str], ds_root: Path, requested_runs: list[str] | None) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        func_dir = ds_root / subject / "func"
        for run in get_run_ids(func_dir, requested_runs):
            yield subject, run


def prepare_subject(
    subject: str,
    ds_root: Path,
    out_root: Path,
    requested_runs: list[str] | None,
    electrode_template: list[str],
    common_electrodes: list[str],
    labels_img: str,
    args: argparse.Namespace,
) -> SubjectPreparationResult:
    eeg_out_dir = out_root / "eeg"
    fmri_out_dir = out_root / "fmri"
    packed_out_dir = out_root / "subjects"
    packed_eeg_samples: list[np.ndarray] = []
    packed_fmri_samples: list[np.ndarray] = []
    packed_labels: list[int] = []
    packed_trial_indices: list[int] = []
    packed_runs: list[str] = []
    packed_sample_ids: list[str] = []
    records: list[SampleRecord] = []
    summaries: list[RunSummary] = []
    eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(args=args)

    for _, run in iter_subject_runs([subject], ds_root, requested_runs):
        func_dir = ds_root / subject / "func"
        eeg_dir = ds_root / subject / "EEG"
        bold_path = func_dir / f"{subject}_task-main_{run}_bold.nii.gz"
        fmri_events_path = func_dir / f"{subject}_task-main_{run}_events.tsv"
        eeg_data_path = eeg_dir / f"EEG_data_{subject}_{run}.mat"
        eeg_events_path = eeg_dir / f"EEG_events_{subject}_{run}.mat"
        if not all(path.exists() for path in [bold_path, fmri_events_path, eeg_data_path, eeg_events_path]):
            continue

        raw_eeg_data, raw_sfreq = load_eeg_data(eeg_data_path, electrode_template=electrode_template, common_electrodes=common_electrodes)
        eeg_data, processed_sfreq = preprocess_eeg(raw_eeg_data, source_sfreq=raw_sfreq, args=args)
        eeg_events = load_eeg_events(eeg_events_path)
        eeg_trials = build_eeg_trial_table(eeg_events)
        fmri_events = load_fmri_events(fmri_events_path, event_type=args.fmri_event_type)
        if fmri_events.empty:
            continue

        if args.fmri_mode == "roi":
            fmri_source, bold_shape, voxel_size = extract_roi_timeseries(
                fmri_nii_path=bold_path,
                labels_img=labels_img,
                tr=args.tr,
                standardize_fmri=args.standardize_fmri,
                include_metadata=True,
            )
        else:
            raw_fmri, bold_shape, voxel_size = load_bold_volume(bold_path, include_metadata=True)
            fmri_source = preprocess_fmri_volume(
                raw_fmri,
                voxel_size=voxel_size,
                source_tr=float(voxel_size[3]),
                target_voxel_size=args.fmri_voxel_size,
                target_tr=float(args.tr),
                max_shape=args.fmri_max_shape,
                use_float16=bool(args.fmri_float16),
            )

        pair_count = min(len(eeg_trials), len(fmri_events))
        if pair_count == 0:
            continue
        eeg_trials = eeg_trials.iloc[:pair_count].reset_index(drop=True)
        fmri_events = fmri_events.iloc[:pair_count].reset_index(drop=True)

        eeg_fmri_offset_sec = estimate_eeg_fmri_offset_sec(eeg_trials=eeg_trials, fmri_events=fmri_events)
        event_counts = {args.fmri_event_type: int(len(fmri_events))}
        summaries.append(
            RunSummary(
                subject=subject,
                run=run,
                bold_shape="x".join(str(dim) for dim in bold_shape),
                voxel_size="x".join(str(dim) for dim in voxel_size),
                target_voxel_size="x".join(str(float(dim)) for dim in args.fmri_voxel_size),
                exported_fmri_shape="x".join(str(dim) for dim in fmri_source.shape),
                eeg_shape="x".join(str(dim) for dim in eeg_data.shape),
                eeg_fmri_offset_sec=eeg_fmri_offset_sec,
                event_counts=str(event_counts),
                eeg_sfreq_hz=processed_sfreq,
                fmri_target_tr_sec=float(args.tr),
                valid_trial_count=int(len(eeg_trials)),
                fmri_trial_count=int(len(fmri_events)),
            )
        )

        eeg_onsets = eeg_trials["eeg_onset_sec"].to_numpy(dtype=np.float64)
        fmri_onsets = fmri_events["onset"].to_numpy(dtype=np.float64)
        for event_index, trial_row in eeg_trials.iterrows():
            trial_type = "dot_stim_validtrials"
            label = int(trial_row["label"])
            label_name = str(trial_row["label_name"])
            eeg_window_length_sec = compute_event_window_sec(
                eeg_onsets,
                event_index=event_index,
                max_window_sec=float(args.eeg_window_sec) if args.eeg_window_sec is not None else float(args.window_sec),
                margin_sec=float(args.window_margin_sec),
                min_window_sec=float(args.min_window_sec),
            )
            fmri_window_length_sec = compute_event_window_sec(
                fmri_onsets,
                event_index=event_index,
                max_window_sec=float(args.fmri_window_sec) if args.fmri_window_sec is not None else float(args.window_sec),
                margin_sec=float(args.window_margin_sec),
                min_window_sec=float(args.min_window_sec),
            )
            if eeg_window_length_sec is None or fmri_window_length_sec is None:
                continue

            eeg_onset_sec = float(trial_row["eeg_onset_sec"])
            fmri_onset_sec = float(fmri_events.iloc[event_index]["onset"])

            try:
                eeg_window = slice_eeg_window(
                    eeg_data,
                    sfreq=processed_sfreq,
                    start_sec=eeg_onset_sec,
                    duration_sec=eeg_window_length_sec,
                )
                if args.eeg_mode == "patched":
                    eeg_window = maybe_patch_eeg(eeg_window, seq_len=eeg_seq_len, patch_len=eeg_patch_len)
                eeg_window = prepare_training_ready_eeg(eeg_window, enabled=bool(args.training_ready))

                if args.fmri_mode == "roi":
                    fmri_window = slice_fmri_window(
                        fmri_source,
                        tr=args.tr,
                        start_sec=fmri_onset_sec,
                        duration_sec=fmri_window_length_sec,
                    )
                else:
                    fmri_window = slice_fmri_volume_window(
                        fmri_source,
                        tr=args.tr,
                        start_sec=fmri_onset_sec,
                        duration_sec=fmri_window_length_sec,
                    )
                fmri_window = prepare_training_ready_fmri(fmri_window, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))
            except ValueError:
                continue

            sample_id = f"{subject}_{run}_trial_{int(trial_row['trial_index']):03d}_{label_name}"

            if args.pack_subject_files:
                packed_eeg_samples.append(eeg_window.astype(np.float32))
                packed_fmri_samples.append(fmri_window.astype(np.float32))
                packed_labels.append(label)
                packed_trial_indices.append(int(trial_row["trial_index"]))
                packed_runs.append(run)
                packed_sample_ids.append(sample_id)
            else:
                eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
                fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
                np.save(eeg_out_path, eeg_window.astype(np.float32))
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
                        window_sec=eeg_window_length_sec,
                        trial_index=int(trial_row["trial_index"]),
                        training_ready=bool(args.training_ready),
                    )
                )

    subject_record: SubjectRecord | None = None
    if args.pack_subject_files and packed_eeg_samples:
        packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
        packed_fmri = stack_subject_samples(packed_fmri_samples, name="fMRI")
        packed_labels_array = np.asarray(packed_labels, dtype=np.int64)
        packed_trial_indices_array = np.asarray(packed_trial_indices, dtype=np.int64)
        packed_runs_array = np.asarray(packed_runs)
        packed_sample_ids_array = np.asarray(packed_sample_ids)

        subject_path = write_subject_memmap_pack(
            packed_out_dir / subject,
            {
                "eeg": packed_eeg,
                "fmri": packed_fmri,
                "labels": packed_labels_array,
                "trial_index": packed_trial_indices_array,
                "run": packed_runs_array,
                "sample_id": packed_sample_ids_array,
            },
        )
        subject_record = SubjectRecord(
            subject=subject,
            subject_path=subject_path.relative_to(out_root).as_posix(),
            sample_count=int(packed_labels_array.shape[0]),
            eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
            fmri_shape="x".join(str(dim) for dim in packed_fmri.shape),
            label_shape="x".join(str(dim) for dim in packed_labels_array.shape),
            training_ready=bool(args.training_ready),
        )

    return SubjectPreparationResult(
        subject=subject,
        records=records,
        subject_record=subject_record,
        summaries=summaries,
    )


def main() -> None:
    args = parse_args()
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    if args.min_window_sec <= 0:
        raise ValueError("--min-window-sec must be positive")

    eeg_window_sec = float(args.eeg_window_sec) if args.eeg_window_sec is not None else float(args.window_sec)
    fmri_window_sec = float(args.fmri_window_sec) if args.fmri_window_sec is not None else float(args.window_sec)
    if eeg_window_sec <= 0 or fmri_window_sec <= 0:
        raise ValueError("--eeg-window-sec and --fmri-window-sec must be positive when provided")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be a positive integer")

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
    electrode_template = load_electrode_template(ds_root)
    common_electrodes = compute_common_electrodes(ds_root, subjects, args.runs, electrode_template)
    save_common_electrode_manifest(out_root, electrode_template, common_electrodes)
    print(f"Using {len(common_electrodes)} shared EEG electrodes across selected files.")
    records: list[SampleRecord] = []
    subject_records: list[SubjectRecord] = []
    summaries: list[RunSummary] = []
    max_workers = min(int(args.num_workers), max(len(subjects), 1))
    if max_workers > 1 and len(subjects) > 1:
        print(f"Preparing ds002739 with {max_workers} worker processes.")
        subject_results: list[SubjectPreparationResult] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_subject = {
                executor.submit(
                    prepare_subject,
                    subject,
                    ds_root,
                    out_root,
                    args.runs,
                    electrode_template,
                    common_electrodes,
                    labels_img,
                    args,
                ): subject
                for subject in subjects
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_subject), total=len(future_to_subject), desc="Preparing ds002739"):
                subject_results.append(future.result())
    else:
        subject_results = [
            prepare_subject(
                subject,
                ds_root,
                out_root,
                args.runs,
                electrode_template,
                common_electrodes,
                labels_img,
                args,
            )
            for subject in tqdm(subjects, desc="Preparing ds002739")
        ]

    subject_order = {subject: index for index, subject in enumerate(subjects)}
    for result in sorted(subject_results, key=lambda item: subject_order[item.subject]):
        records.extend(result.records)
        summaries.extend(result.summaries)
        if result.subject_record is not None:
            subject_records.append(result.subject_record)

    if not records and not subject_records:
        raise RuntimeError("No samples were exported. Check subject IDs, run IDs, event definitions, and window settings.")

    if subject_records:
        pd.DataFrame(record.__dict__ for record in sorted(subject_records, key=lambda item: item.subject)).to_csv(out_root / "manifest_all.csv", index=False)
    else:
        pd.DataFrame(record.__dict__ for record in sorted(records, key=lambda item: item.sample_id)).to_csv(out_root / "manifest_all.csv", index=False)
    pd.DataFrame(summary.__dict__ for summary in sorted(summaries, key=lambda item: (item.subject, item.run))).to_csv(out_root / "run_summary.csv", index=False)

    manifest_path = out_root / "manifest_all.csv"
    if args.split_mode == "subject":
        split_output_dir = args.split_output_dir.resolve() if args.split_output_dir is not None else (out_root / "splits_subjectwise")
        write_subject_splits(
            manifest_path=manifest_path,
            output_dir=split_output_dir,
            train_subjects=int(args.train_subjects),
            val_subjects=int(args.val_subjects),
            test_subjects=int(args.test_subjects),
        )
    elif args.split_mode == "loso":
        split_output_dir = args.split_output_dir.resolve() if args.split_output_dir is not None else (out_root / "loso_subjectwise")
        write_loso_splits(
            manifest_path=manifest_path,
            output_dir=split_output_dir,
            val_subjects=int(args.val_subjects),
        )


if __name__ == "__main__":
    main()