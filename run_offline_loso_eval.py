from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mmcontrast.visualization import save_confusion_matrix


DATASET_CLASS_NAMES = {
    "ds009999": ["negative", "neutral", "positive"],
}

DATASET_CONFUSION_TITLES = {
    "ds002336": "XP1",
    "ds002338": "XP2",
    "ds009999": "SEED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Offline LOSO checkpoint evaluation")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name used in output file names.")
    parser.add_argument("--config", type=str, required=True, help="Finetune config path.")
    parser.add_argument("--checkpoints-root", type=str, required=True, help="Root directory that contains fold_*/checkpoints/best.pth.")
    parser.add_argument("--checkpoint-relpath", type=str, default="checkpoints/best.pth", help="Checkpoint path relative to each fold directory.")
    parser.add_argument("--split-root", type=str, default="", help="Optional LOSO split root. Defaults to <root_dir>/loso_subjectwise from config.")
    parser.add_argument("--root-dir", type=str, default="", help="Optional dataset cache root override.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for per-fold test artifacts and LOSO summaries.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional num_workers override.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Optional comma-separated class names for confusion matrix axes. Defaults to built-ins or numeric labels.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float("nan") if not valid else sum(valid) / len(valid)


def std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) <= 1:
        return 0.0
    avg = sum(valid) / len(valid)
    return math.sqrt(sum((value - avg) ** 2 for value in valid) / (len(valid) - 1))


def resolve_class_names(dataset_name: str, raw_class_names: str, num_classes: int) -> list[str]:
    if raw_class_names.strip():
        values = [item.strip() for item in raw_class_names.split(",") if item.strip()]
    else:
        values = list(DATASET_CLASS_NAMES.get(dataset_name.strip().lower(), []))
    if len(values) < num_classes:
        values = values + [str(index) for index in range(len(values), num_classes)]
    return values[:num_classes]


def resolve_confusion_title(dataset_name: str, default_title: str) -> str:
    return DATASET_CONFUSION_TITLES.get(dataset_name.strip().lower(), default_title)


def remove_per_fold_confusion_artifacts(fold_output_dir: Path) -> None:
    for filename in ("test_confusion_matrix.png", "test_confusion_matrix.svg", "test_confusion_matrix.json"):
        path = fold_output_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    data_cfg = dict(cfg.get("data", {}) or {})
    root_dir_value = args.root_dir.strip() or str(data_cfg.get("root_dir", "")).strip()
    if not root_dir_value:
        raise ValueError("Could not resolve data.root_dir from config. Pass --root-dir explicitly.")
    root_dir = Path(root_dir_value)
    if not root_dir.is_absolute():
        root_dir = (repo_root / root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root_dir not found: {root_dir}")

    split_root = Path(args.split_root).resolve() if args.split_root.strip() else (root_dir / "loso_subjectwise").resolve()
    if not split_root.exists():
        raise FileNotFoundError(f"LOSO split root not found: {split_root}")

    checkpoints_root = Path(args.checkpoints_root)
    if not checkpoints_root.is_absolute():
        checkpoints_root = (repo_root / checkpoints_root).resolve()
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root not found: {checkpoints_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir.strip() else (repo_root / "outputs" / f"{args.dataset_name}_offline_eval").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(path for path in split_root.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found under {split_root}")

    summary_rows: list[dict[str, Any]] = []
    loso_labels: list[int] = []
    loso_preds: list[int] = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        checkpoint_path = checkpoints_root / fold_name / args.checkpoint_relpath
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {fold_name}: {checkpoint_path}")

        fold_output_dir = output_dir / fold_name
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(repo_root / "run_finetune.py"),
            "--config",
            str(config_path),
            "--train-manifest",
            str((fold_dir / "manifest_train.csv").resolve()),
            "--val-manifest",
            str((fold_dir / "manifest_val.csv").resolve()),
            "--test-manifest",
            str((fold_dir / "manifest_test.csv").resolve()),
            "--root-dir",
            str(root_dir),
            "--output-dir",
            str(fold_output_dir),
            "--finetune-checkpoint",
            str(checkpoint_path),
            "--test-only",
        ]
        if args.eval_batch_size is not None:
            command.extend(["--eval-batch-size", str(args.eval_batch_size)])
        if args.num_workers is not None:
            command.extend(["--num-workers", str(args.num_workers)])
        if args.force_cpu:
            command.append("--force-cpu")

        print(f"[{fold_name}] offline test", flush=True)
        completed = subprocess.run(command, cwd=str(repo_root), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Offline evaluation failed for {fold_name} with exit code {completed.returncode}")

        metrics_path = fold_output_dir / "test_metrics.json"
        logits_path = fold_output_dir / "test_logits.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing test_metrics.json after evaluation: {metrics_path}")
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing test_logits.csv after evaluation: {logits_path}")
        remove_per_fold_confusion_artifacts(fold_output_dir)

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle) or {}
        logits_df = pd.read_csv(logits_path)
        labels = logits_df["label"].astype(int).tolist()
        preds = logits_df["pred"].astype(int).tolist()
        loso_labels.extend(labels)
        loso_preds.extend(preds)

        summary_rows.append(
            {
                "fold": fold_name,
                "checkpoint_path": str(checkpoint_path),
                "accuracy": float(metrics.get("accuracy", float("nan"))),
                "accuracy_std": float(metrics.get("accuracy_std", float("nan"))),
                "macro_f1": float(metrics.get("macro_f1", float("nan"))),
                "macro_f1_std": float(metrics.get("macro_f1_std", float("nan"))),
                "loss": float(metrics.get("loss", float("nan"))),
                "sample_count": int(len(labels)),
            }
        )

    if summary_rows:
        summary_rows.append(
            {
                "fold": "CROSS_FOLD_MEAN_STD",
                "checkpoint_path": "",
                "accuracy": mean([float(row["accuracy"]) for row in summary_rows]),
                "accuracy_std": std([float(row["accuracy"]) for row in summary_rows]),
                "macro_f1": mean([float(row["macro_f1"]) for row in summary_rows]),
                "macro_f1_std": std([float(row["macro_f1"]) for row in summary_rows]),
                "loss": mean([float(row["loss"]) for row in summary_rows]),
                "sample_count": int(sum(int(row["sample_count"]) for row in summary_rows)),
            }
        )

    if loso_labels and loso_preds:
        num_classes = int(max(loso_labels + loso_preds) + 1)
        class_names = resolve_class_names(args.dataset_name, args.class_names, num_classes)
        report = save_confusion_matrix(
            labels=np.asarray(loso_labels, dtype=int),
            preds=np.asarray(loso_preds, dtype=int),
            output_path=output_dir / f"confusion_matrix_{args.dataset_name}_loso.png",
            class_names=class_names,
            title=resolve_confusion_title(args.dataset_name, f"{args.dataset_name} LOSO Confusion Matrix"),
            normalize=False,
        )
        with (output_dir / f"confusion_matrix_{args.dataset_name}_loso.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    summary_path = output_dir / "loso_test_summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Saved LOSO offline evaluation outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
