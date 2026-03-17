from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

JOINT_CHANNEL_MANIFEST = Path(__file__).resolve().parents[2] / "cache" / "joint_contrastive" / "eeg_channels_target.csv"


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def _load_channel_names_from_manifest(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names: list[str] = []
        for row in reader:
            name = str(row.get("target_channel_name", "")).strip()
            if name:
                names.append(name)
    return names


def resolve_current_channel_manifest(data_cfg: dict[str, Any]) -> Path | None:
    manifest_path = str(data_cfg.get("eeg_channel_target_manifest", "")).strip()
    if manifest_path:
        resolved = Path(manifest_path)
        return resolved if resolved.exists() else None
    root_dir = str(data_cfg.get("root_dir", "")).strip()
    if not root_dir:
        return None
    candidate = Path(root_dir) / "eeg_channels_target.csv"
    return candidate if candidate.exists() else None


def build_eeg_channel_summary(data_cfg: dict[str, Any], fallback_raw_count: int | None = None) -> str | None:
    current_manifest = resolve_current_channel_manifest(data_cfg)
    current_names = _load_channel_names_from_manifest(current_manifest) if current_manifest is not None else []
    raw_total = len(current_names) if current_names else fallback_raw_count
    if raw_total is None:
        return None

    summary_parts = [f"EEG raw channels={raw_total}."]
    common_names = _load_channel_names_from_manifest(JOINT_CHANNEL_MANIFEST) if JOINT_CHANNEL_MANIFEST.exists() else []
    if current_names and common_names:
        current_lookup = {_normalize_channel_name(name) for name in current_names}
        matched_common = sum(1 for name in common_names if _normalize_channel_name(name) in current_lookup)
        summary_parts.append(f"Common-channel overlap={matched_common}/{len(common_names)}.")
    return " ".join(summary_parts)
