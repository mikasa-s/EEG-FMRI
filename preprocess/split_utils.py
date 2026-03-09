from __future__ import annotations

from pathlib import Path

import pandas as pd


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
