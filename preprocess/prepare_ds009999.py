from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm

from preprocess_common import (
    add_dataset_io_args,
    add_subject_args,
    add_subject_packing_and_split_args,
    add_training_ready_arg,
    build_canonical_subject_map,
    load_target_channel_names,
    make_channel_metadata_rows,
    make_subject_uid,
    prepare_training_ready_eeg,
    reorder_eeg_channels,
    stack_subject_samples,
    write_channel_metadata,
    write_loso_splits,
    write_subject_mapping,
    write_subject_memmap_pack,
    write_subject_splits,
)


DATASET_NAME = "ds009999"
SEED_CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
    "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
LABEL_NAME_MAP = {-1: "negative", 0: "neutral", 1: "positive"}


@dataclass(frozen=True)
class SubjectRecord:
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    subject_path: str
    sample_count: int
    eeg_shape: str
    label_shape: str
    training_ready: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SEED emotion EEG data as ds009999 for EEG-only finetuning.")
    add_dataset_io_args(parser, ds_root_help="Path to the SEED root.", output_root_help="Output cache directory.")
    add_subject_args(parser, subject_example="sub01")
    parser.add_argument(
        "--labels-mat",
        type=Path,
        default=None,
        help="Path to SEED label.mat. If omitted, the script tries <ds-root>/label.mat and <ds-root>/Preprocessed_EEG/label.mat.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Optional session filters. Matches either parent directory name or the inferred session token from file names.",
    )
    parser.add_argument("--input-sfreq", type=float, default=200.0, help="Sampling rate of the source EEG arrays.")
    parser.add_argument("--eeg-target-sfreq", type=float, default=200.0, help="Target EEG sampling rate.")
    parser.add_argument("--eeg-lfreq", type=float, default=0.5, help="Low cutoff for EEG band-pass filter.")
    parser.add_argument("--eeg-hfreq", type=float, default=40.0, help="High cutoff for EEG band-pass filter.")
    parser.add_argument("--window-sec", type=float, default=8.0, help="Sliding window length in seconds.")
    parser.add_argument("--window-overlap-sec", type=float, default=0.0, help="Adjacent EEG window overlap in seconds.")
    parser.add_argument("--eeg-mode", choices=["continuous", "patched"], default="patched", help="continuous saves [C,T]; patched saves [C,S,P].")
    parser.add_argument("--eeg-seq-len", type=int, default=8, help="Sequence length S when --eeg-mode=patched.")
    parser.add_argument("--eeg-patch-len", type=int, default=200, help="Patch length P when --eeg-mode=patched.")
    parser.add_argument("--target-channel-manifest", type=Path, default=None, help="Optional target EEG channel manifest.")
    add_subject_packing_and_split_args(
        parser,
        pack_help="Pack each subject into a subject-level memmap directory.",
        split_help="Split exported data by subject or LOSO.",
        train_subjects=12,
        val_subjects=2,
        test_subjects=1,
    )
    add_training_ready_arg(parser)
    parser.set_defaults(training_ready=True)
    return parser.parse_args()


def infer_subject_and_session(mat_path: Path) -> tuple[str, str]:
    stem = mat_path.stem
    digit_groups = re.findall(r"\d+", stem)
    if not digit_groups:
        digit_groups = re.findall(r"\d+", mat_path.parent.name)
    if not digit_groups:
        raise ValueError(f"Could not infer SEED subject id from file name: {mat_path}")
    subject_id = f"sub{int(digit_groups[0]):02d}"
    if len(digit_groups) >= 2:
        session_id = f"ses{int(digit_groups[1]):02d}"
    else:
        parent_name = mat_path.parent.name.strip()
        session_id = parent_name if parent_name and parent_name != mat_path.parent.parent.name else stem
    return subject_id, session_id


def find_seed_mat_files(ds_root: Path, requested_subjects: list[str] | None, requested_sessions: list[str] | None, labels_mat: Path) -> list[Path]:
    subject_filter = {value.strip() for value in requested_subjects or []}
    session_filter = {value.strip() for value in requested_sessions or []}
    files: list[Path] = []
    label_names = {"label", "labels"}
    data_root = ds_root / "Preprocessed_EEG"
    search_root = data_root if data_root.exists() else ds_root
    for mat_path in sorted(search_root.rglob("*.mat")):
        if mat_path.resolve() == labels_mat.resolve():
            continue
        if mat_path.stem.strip().lower() in label_names:
            continue
        if "extractedfeatures" in {part.lower() for part in mat_path.parts}:
            continue
        subject_id, session_id = infer_subject_and_session(mat_path)
        if subject_filter and subject_id not in subject_filter:
            continue
        if session_filter and session_id not in session_filter and mat_path.parent.name not in session_filter:
            continue
        files.append(mat_path)
    return files


def load_seed_labels(labels_mat_path: Path) -> tuple[list[int], dict[int, int]]:
    payload = loadmat(labels_mat_path)
    label_values: np.ndarray | None = None
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value).squeeze()
        if array.ndim == 1 and array.size > 0:
            label_values = array.astype(np.int64)
            break
    if label_values is None:
        raise ValueError(f"Could not find 1D labels array in {labels_mat_path}")
    raw_labels = [int(item) for item in label_values.tolist()]
    unique_labels = sorted(set(raw_labels))
    label_map = {raw_label: mapped for mapped, raw_label in enumerate(unique_labels)}
    return raw_labels, label_map


def normalize_eeg_array(array: np.ndarray, channel_count: int) -> np.ndarray:
    eeg = np.asarray(array, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG trial array [C,T] or [T,C], got {eeg.shape}")
    if eeg.shape[0] == channel_count:
        return eeg
    if eeg.shape[1] == channel_count:
        return eeg.T.copy()
    raise ValueError(f"Cannot align EEG trial shape {eeg.shape} to channel count {channel_count}")


def extract_trial_entries(mat_path: Path, channel_count: int) -> list[tuple[int, np.ndarray]]:
    payload = loadmat(mat_path)
    trial_entries: list[tuple[int, np.ndarray]] = []
    fallback_index = 0
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.dtype.kind not in {"f", "i", "u"} or array.ndim != 2:
            continue
        try:
            eeg = normalize_eeg_array(array, channel_count=channel_count)
        except ValueError:
            continue
        match = re.search(r"(\d+)$", key)
        trial_index = int(match.group(1)) if match else fallback_index + 1
        fallback_index += 1
        trial_entries.append((trial_index, eeg))
    if not trial_entries:
        raise ValueError(f"No EEG trial arrays found in {mat_path}")
    trial_entries.sort(key=lambda item: item[0])
    return trial_entries


def maybe_resample_eeg(eeg: np.ndarray, input_sfreq: float, target_sfreq: float) -> np.ndarray:
    if abs(float(input_sfreq) - float(target_sfreq)) < 1e-6:
        return eeg.astype(np.float32, copy=False)
    target_points = int(round(eeg.shape[-1] * float(target_sfreq) / float(input_sfreq)))
    return resample(eeg, target_points, axis=-1).astype(np.float32)


def maybe_filter_eeg(eeg: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    filter_input = np.asarray(eeg, dtype=np.float64)
    return mne.filter.filter_data(
        filter_input,
        sfreq=float(sfreq),
        l_freq=float(l_freq),
        h_freq=float(h_freq),
        verbose="ERROR",
    ).astype(np.float32)


def build_fixed_windows(eeg: np.ndarray, window_points: int, step_points: int) -> list[np.ndarray]:
    total_points = int(eeg.shape[-1])
    if total_points <= window_points:
        padded = np.zeros((eeg.shape[0], window_points), dtype=np.float32)
        padded[:, :total_points] = eeg[:, :total_points]
        return [padded]
    starts = list(range(0, total_points - window_points + 1, step_points))
    final_start = total_points - window_points
    if starts[-1] != final_start:
        starts.append(final_start)
    return [eeg[:, start : start + window_points].astype(np.float32, copy=False) for start in starts]


def format_eeg_sample(eeg_window: np.ndarray, eeg_mode: str, eeg_seq_len: int, eeg_patch_len: int, training_ready: bool) -> np.ndarray:
    output = prepare_training_ready_eeg(eeg_window, enabled=training_ready)
    if eeg_mode == "continuous":
        return output.astype(np.float32, copy=False)
    return output.reshape(output.shape[0], eeg_seq_len, eeg_patch_len).astype(np.float32, copy=False)


def resolve_labels_mat(ds_root: Path, configured_path: Path | None) -> Path:
    if configured_path is not None:
        return configured_path.resolve()
    candidates = [
        ds_root / "label.mat",
        ds_root / "Preprocessed_EEG" / "label.mat",
        ds_root / "ExtractedFeatures" / "label.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def main() -> None:
    args = parse_args()
    ds_root = args.ds_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if not bool(args.pack_subject_files):
        raise ValueError("prepare_ds009999.py currently requires --pack-subject-files.")

    labels_mat = resolve_labels_mat(ds_root, args.labels_mat)
    if not labels_mat.exists():
        raise FileNotFoundError(f"SEED labels file not found: {labels_mat}")
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    if args.window_overlap_sec < 0 or args.window_overlap_sec >= args.window_sec:
        raise ValueError("--window-overlap-sec must satisfy 0 <= overlap < window")
    if args.eeg_seq_len <= 0 or args.eeg_patch_len <= 0:
        raise ValueError("--eeg-seq-len and --eeg-patch-len must be positive")

    raw_labels, label_map = load_seed_labels(labels_mat)
    target_channel_names = load_target_channel_names(args.target_channel_manifest) if args.target_channel_manifest else list(SEED_CHANNEL_NAMES)
    source_channel_names = list(SEED_CHANNEL_NAMES)
    mat_files = find_seed_mat_files(ds_root, args.subjects, args.sessions, labels_mat)
    if not mat_files:
        raise RuntimeError("No SEED .mat files found. Check --ds-root, --subjects, and --sessions.")

    subject_ids = sorted({infer_subject_and_session(path)[0] for path in mat_files})
    canonical_subject_map = build_canonical_subject_map(subject_ids, min_digits=2)
    dataset_channel_rows = make_channel_metadata_rows(DATASET_NAME, source_channel_names)
    subject_mapping_rows = [
        {
            "dataset": DATASET_NAME,
            "source_subject": subject_id,
            "subject": canonical_subject_map[subject_id],
            "subject_uid": make_subject_uid(DATASET_NAME, canonical_subject_map[subject_id]),
        }
        for subject_id in subject_ids
    ]
    channel_mapping_rows: list[dict[str, object]] = []

    window_points = int(round(float(args.window_sec) * float(args.eeg_target_sfreq)))
    expected_points = int(args.eeg_seq_len) * int(args.eeg_patch_len)
    if args.eeg_mode == "patched" and window_points != expected_points:
        raise ValueError(f"Patched EEG expects {window_points} points per window, but eeg_seq_len * eeg_patch_len = {expected_points}")
    step_points = int(round((float(args.window_sec) - float(args.window_overlap_sec)) * float(args.eeg_target_sfreq)))
    packed_root = out_root / "subjects"
    subject_records: list[SubjectRecord] = []

    files_by_subject: dict[str, list[Path]] = {}
    for mat_path in mat_files:
        subject_id, _ = infer_subject_and_session(mat_path)
        files_by_subject.setdefault(subject_id, []).append(mat_path)

    for subject_id in tqdm(sorted(files_by_subject.keys()), desc=f"Preparing {DATASET_NAME}"):
        canonical_subject = canonical_subject_map[subject_id]
        subject_uid = make_subject_uid(DATASET_NAME, canonical_subject)
        packed_eeg_samples: list[np.ndarray] = []
        packed_labels: list[int] = []
        packed_sample_ids: list[str] = []
        packed_sessions: list[str] = []
        packed_trial_indices: list[int] = []
        packed_label_names: list[str] = []

        for mat_path in sorted(files_by_subject[subject_id]):
            _, session_id = infer_subject_and_session(mat_path)
            for trial_index, trial_eeg in extract_trial_entries(mat_path, channel_count=len(source_channel_names)):
                if trial_index <= 0 or trial_index > len(raw_labels):
                    raise ValueError(f"Trial index {trial_index} in {mat_path} is outside label range 1..{len(raw_labels)}")
                raw_label = int(raw_labels[trial_index - 1])
                mapped_label = int(label_map[raw_label])
                label_name = LABEL_NAME_MAP.get(raw_label, f"label_{raw_label}")
                eeg = maybe_resample_eeg(trial_eeg, input_sfreq=float(args.input_sfreq), target_sfreq=float(args.eeg_target_sfreq))
                eeg = maybe_filter_eeg(eeg, sfreq=float(args.eeg_target_sfreq), l_freq=float(args.eeg_lfreq), h_freq=float(args.eeg_hfreq))
                eeg, mapping_rows = reorder_eeg_channels(eeg, source_channel_names, target_channel_names)
                if not channel_mapping_rows:
                    channel_mapping_rows.extend({"dataset": DATASET_NAME, **row} for row in mapping_rows)

                windows = build_fixed_windows(eeg, window_points=window_points, step_points=step_points)
                for window_index, eeg_window in enumerate(windows, start=1):
                    sample_id = f"{DATASET_NAME}_{canonical_subject}_{session_id}_trial-{trial_index:02d}_win-{window_index:02d}"
                    packed_eeg_samples.append(
                        format_eeg_sample(
                            eeg_window,
                            eeg_mode=args.eeg_mode,
                            eeg_seq_len=int(args.eeg_seq_len),
                            eeg_patch_len=int(args.eeg_patch_len),
                            training_ready=bool(args.training_ready),
                        )
                    )
                    packed_labels.append(mapped_label)
                    packed_sample_ids.append(sample_id)
                    packed_sessions.append(session_id)
                    packed_trial_indices.append(trial_index)
                    packed_label_names.append(label_name)

        if not packed_eeg_samples:
            continue

        packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
        packed_labels_array = np.asarray(packed_labels, dtype=np.int64)
        arrays_to_write = {
            "eeg": packed_eeg,
            "labels": packed_labels_array,
            "sample_id": np.asarray(packed_sample_ids),
            "session": np.asarray(packed_sessions),
            "trial_index": np.asarray(packed_trial_indices, dtype=np.int64),
            "label_name": np.asarray(packed_label_names),
        }
        subject_path = write_subject_memmap_pack(packed_root / subject_uid, arrays_to_write)
        subject_records.append(
            SubjectRecord(
                dataset=DATASET_NAME,
                subject=canonical_subject,
                subject_uid=subject_uid,
                original_subject=subject_id,
                subject_path=subject_path.relative_to(out_root).as_posix(),
                sample_count=int(packed_labels_array.shape[0]),
                eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
                label_shape="x".join(str(dim) for dim in packed_labels_array.shape),
                training_ready=bool(args.training_ready),
            )
        )

    if not subject_records:
        raise RuntimeError("No SEED samples were exported.")

    pd.DataFrame(record.__dict__ for record in subject_records).to_csv(out_root / "manifest_all.csv", index=False)
    write_subject_mapping(subject_mapping_rows, out_root / "subject_mapping.csv")
    write_channel_metadata(dataset_channel_rows, out_root / "eeg_channels_dataset.csv")
    write_channel_metadata(
        [{"target_channel_index": index, "target_channel_name": name} for index, name in enumerate(target_channel_names)],
        out_root / "eeg_channels_target.csv",
    )
    write_channel_metadata(channel_mapping_rows, out_root / "eeg_channel_mapping.csv")

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

    print(
        f"Prepared {DATASET_NAME}: subjects={len(subject_records)}, files={len(mat_files)}, "
        f"training_ready={bool(args.training_ready)}"
    )


if __name__ == "__main__":
    main()
