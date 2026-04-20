from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


FULL_MODE = "full"
STRICT_MODE = "strict"
VALID_PRETRAIN_MODES = {FULL_MODE, STRICT_MODE}
DEFAULT_FULL_CACHE_DIR = Path("cache/joint_contrastive")
DEFAULT_FULL_OUTPUT_ROOT = Path("pretrained_weights/full")
DEFAULT_STRICT_OUTPUT_ROOT = Path("pretrained_weights/strict")


def normalize_pretrain_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_PRETRAIN_MODES:
        raise ValueError(f"Unsupported pretrain mode '{mode}'. Expected one of: {sorted(VALID_PRETRAIN_MODES)}")
    return normalized


def normalize_subject_token(subject: str) -> str:
    normalized = str(subject).strip().replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("Held-out subject cannot be empty.")
    return normalized


def require_strict_identifiers(mode: str, target_dataset: str, held_out_subject: str) -> tuple[str, str]:
    normalized_mode = normalize_pretrain_mode(mode)
    if normalized_mode != STRICT_MODE:
        return "", ""
    dataset = str(target_dataset).strip()
    subject = normalize_subject_token(held_out_subject)
    if not dataset:
        raise ValueError("strict pretrain mode requires a non-empty target dataset.")
    return dataset, subject


def infer_pretrain_objective_name(config_path: str | Path, cfg: dict[str, Any] | None = None) -> str:
    config_stem = Path(config_path).stem.strip().lower()
    if config_stem.startswith("train_joint_"):
        suffix = config_stem[len("train_joint_") :].strip()
        if suffix:
            return suffix
    objective = str(((cfg or {}).get("train", {}) or {}).get("pretrain_objective", "")).strip().lower()
    if objective == "shared_private":
        return "contrastive"
    if objective:
        return objective
    return "contrastive"


def resolve_pretrain_cache_dir(*, project_root: Path, cache_root: str = "") -> Path:
    base = Path(cache_root) if str(cache_root).strip() else DEFAULT_FULL_CACHE_DIR
    return base if base.is_absolute() else (project_root / base).resolve()


def resolve_pretrain_output_dir(
    *,
    project_root: Path,
    mode: str,
    objective_name: str,
    target_dataset: str = "",
    held_out_subject: str = "",
    output_root: str = "",
) -> Path:
    normalized_mode = normalize_pretrain_mode(mode)
    objective = str(objective_name).strip() or "contrastive"
    if normalized_mode == FULL_MODE:
        base = Path(output_root) if str(output_root).strip() else DEFAULT_FULL_OUTPUT_ROOT
        resolved_base = base if base.is_absolute() else (project_root / base).resolve()
        return (resolved_base / objective).resolve()
    dataset, subject = require_strict_identifiers(normalized_mode, target_dataset, held_out_subject)
    base = Path(output_root) if str(output_root).strip() else DEFAULT_STRICT_OUTPUT_ROOT
    resolved_base = base if base.is_absolute() else (project_root / base).resolve()
    return (resolved_base / dataset / subject / objective).resolve()


def resolve_pretrain_checkpoint_path(
    *,
    project_root: Path,
    mode: str,
    objective_name: str,
    target_dataset: str = "",
    held_out_subject: str = "",
    output_root: str = "",
    checkpoint_relpath: str = "checkpoints/best.pth",
) -> Path:
    return (
        resolve_pretrain_output_dir(
            project_root=project_root,
            mode=mode,
            objective_name=objective_name,
            target_dataset=target_dataset,
            held_out_subject=held_out_subject,
            output_root=output_root,
        )
        / checkpoint_relpath
    ).resolve()


def infer_target_dataset_from_root_dir(root_dir_value: str) -> str:
    return Path(str(root_dir_value).strip()).name if str(root_dir_value).strip() else ""


def infer_held_out_subject_from_fold_name(fold_name: str) -> str:
    name = str(fold_name).strip()
    if name.startswith("fold_"):
        return name[len("fold_") :]
    return name


def infer_held_out_subject_from_manifest(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values: list[str] = []
        for row in reader:
            subject_uid = str(row.get("subject_uid", "")).strip()
            subject = str(row.get("subject", "")).strip()
            value = subject_uid or subject
            if value:
                values.append(value)
        unique_values = list(dict.fromkeys(values))
    if not unique_values:
        raise ValueError(f"Could not infer held-out subject from manifest: {path}")
    if len(unique_values) != 1:
        raise ValueError(f"Expected exactly one held-out subject in test manifest {path}, got {unique_values}")
    return unique_values[0]


def create_strict_manifest(
    *,
    source_manifest_path: Path,
    output_manifest_path: Path,
    target_dataset: str,
    held_out_subject: str,
) -> dict[str, Any]:
    dataset = str(target_dataset).strip()
    subject_token = normalize_subject_token(held_out_subject)
    manifest = pd.read_csv(source_manifest_path)
    if "dataset" not in manifest.columns:
        raise ValueError(f"Manifest missing required 'dataset' column: {source_manifest_path}")

    candidate_columns = [column for column in ("subject_uid", "original_subject", "subject") if column in manifest.columns]
    if not candidate_columns:
        raise ValueError(f"Manifest does not contain any subject identifier columns: {source_manifest_path}")

    subject_values = {subject_token}
    if "_" in subject_token:
        subject_values.add(subject_token.split("_", 1)[1])
    exclusion_mask = manifest["dataset"].astype(str) == dataset
    subject_mask = pd.Series(False, index=manifest.index)
    for column in candidate_columns:
        column_values = manifest[column].astype(str).str.strip()
        subject_mask = subject_mask | column_values.isin(subject_values)
    filtered = manifest.loc[~(exclusion_mask & subject_mask)].copy()
    removed_count = int(len(manifest) - len(filtered))
    if removed_count <= 0:
        raise ValueError(
            f"Strict pretrain filtering removed zero rows from {source_manifest_path}. "
            f"dataset={dataset}, held_out_subject={held_out_subject}, matched_columns={candidate_columns}"
        )

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_manifest_path, index=False)
    return {
        "source_manifest": str(source_manifest_path),
        "output_manifest": str(output_manifest_path),
        "target_dataset": dataset,
        "held_out_subject": held_out_subject,
        "matched_columns": candidate_columns,
        "rows_before": int(len(manifest)),
        "rows_after": int(len(filtered)),
        "rows_removed": removed_count,
    }


def list_dataset_subjects_from_manifest(source_manifest_path: Path, target_dataset: str) -> list[str]:
    dataset = str(target_dataset).strip()
    manifest = pd.read_csv(source_manifest_path)
    if "dataset" not in manifest.columns:
        raise ValueError(f"Manifest missing required 'dataset' column: {source_manifest_path}")
    dataset_rows = manifest.loc[manifest["dataset"].astype(str) == dataset].copy()
    if dataset_rows.empty:
        raise ValueError(f"No rows found for dataset={dataset} in manifest {source_manifest_path}")

    for column in ("subject_uid", "original_subject", "subject"):
        if column in dataset_rows.columns:
            values = [str(value).strip() for value in dataset_rows[column].dropna().tolist() if str(value).strip()]
            unique_values = list(dict.fromkeys(values))
            if unique_values:
                return unique_values
    raise ValueError(f"No subject identifier columns with values found for dataset={dataset} in manifest {source_manifest_path}")
