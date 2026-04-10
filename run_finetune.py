from __future__ import annotations

"""分类微调入口脚本。"""

import argparse
import copy
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import yaml

from mmcontrast.finetune_runner import run_finetuning

PROJECT_ROOT = Path(__file__).resolve().parent
# 允许直接从项目根目录执行脚本，而不依赖外部 PYTHONPATH 配置。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析微调所需参数，支持直接覆盖常用配置。"""
    parser = argparse.ArgumentParser("EEG-fMRI finetuning for classification")
    parser.add_argument("--config", type=str, default="configs/finetune_ds002739.yaml")
    parser.add_argument("--loso", action="store_true", help="Run all fold_* under LOSO split root and write loso_finetune_summary.csv.")
    parser.add_argument("--split-root", type=str, default="", help="Optional LOSO split root. Defaults to <root_dir>/loso_subjectwise.")
    parser.add_argument("--train-manifest", type=str, default="")
    parser.add_argument("--val-manifest", type=str, default="")
    parser.add_argument("--test-manifest", type=str, default="")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--contrastive-checkpoint", type=str, default="")
    parser.add_argument("--finetune-checkpoint", type=str, default="")
    parser.add_argument("--selection-metric", type=str, default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eeg-baseline-category", type=str, default="")
    parser.add_argument("--eeg-baseline-model", type=str, default="")
    parser.add_argument("--eeg-baseline-load-pretrained", type=str, default="")
    parser.add_argument("--eeg-baseline-checkpoint", type=str, default="")
    parser.add_argument("--classifier-mode", type=str, default="")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Generic override in the form section.key=value. Can be repeated.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assign_nested_value(payload: dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    def parse_bool_text(raw: str) -> bool:
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw}")

    mapping: list[tuple[str, object]] = [
        ("data.train_manifest_csv", args.train_manifest.strip()),
        ("data.val_manifest_csv", args.val_manifest.strip()),
        ("data.test_manifest_csv", args.test_manifest.strip()),
        ("data.root_dir", args.root_dir.strip()),
        ("finetune.output_dir", args.output_dir.strip()),
        ("finetune.contrastive_checkpoint_path", args.contrastive_checkpoint.strip()),
        ("finetune.eval_checkpoint_path", args.finetune_checkpoint.strip()),
        ("finetune.selection_metric", args.selection_metric.strip()),
        ("finetune.epochs", args.epochs),
        ("train.batch_size", args.batch_size),
        ("train.eval_batch_size", args.eval_batch_size),
        ("train.num_workers", args.num_workers),
        ("finetune.lr", args.lr),
    ]

    for dotted_key, value in mapping:
        if value not in (None, ""):
            assign_nested_value(config, dotted_key, value)

    if args.eeg_baseline_category.strip() or args.eeg_baseline_model.strip():
        assign_nested_value(config, "finetune.eeg_baseline.enabled", True)
    if args.eeg_baseline_category.strip():
        assign_nested_value(config, "finetune.eeg_baseline.category", args.eeg_baseline_category.strip())
    if args.eeg_baseline_model.strip():
        assign_nested_value(config, "finetune.eeg_baseline.model_name", args.eeg_baseline_model.strip())
    if args.eeg_baseline_load_pretrained.strip():
        assign_nested_value(
            config,
            "finetune.eeg_baseline.load_pretrained_weights",
            parse_bool_text(args.eeg_baseline_load_pretrained),
        )
    if args.eeg_baseline_checkpoint.strip():
        assign_nested_value(config, "finetune.eeg_baseline.checkpoint_path", args.eeg_baseline_checkpoint.strip())
    if args.classifier_mode.strip():
        assign_nested_value(config, "finetune.classifier_mode", args.classifier_mode.strip())

    if args.force_cpu:
        assign_nested_value(config, "train.force_cpu", True)

    if args.test_only:
        assign_nested_value(config, "finetune.test_only", True)

    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected section.key=value")
        dotted_key, raw_value = override.split("=", 1)
        assign_nested_value(config, dotted_key.strip(), yaml.safe_load(raw_value))

    return config


def write_runtime_config(config: dict, source_config: Path) -> Path:
    runtime_dir = PROJECT_ROOT / ".runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix=f"{source_config.stem}_",
        dir=runtime_dir,
        delete=False,
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        return Path(handle.name)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float("nan") if not valid else sum(valid) / len(valid)


def _std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) <= 1:
        return 0.0
    avg = sum(valid) / len(valid)
    return math.sqrt(sum((value - avg) ** 2 for value in valid) / (len(valid) - 1))


def write_loso_summary(finetune_root: Path) -> Path:
    rows: list[dict[str, float | str]] = []
    for fold_dir in sorted(path for path in finetune_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics_path = fold_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle) or {}
        rows.append(
            {
                "fold": fold_dir.name,
                "accuracy": float(metrics.get("accuracy", float("nan"))),
                "accuracy_std": float(metrics.get("accuracy_std", float("nan"))),
                "macro_f1": float(metrics.get("macro_f1", float("nan"))),
                "macro_f1_std": float(metrics.get("macro_f1_std", float("nan"))),
                "loss": float(metrics.get("loss", float("nan"))),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No fold test_metrics.json files found under {finetune_root}")

    rows.append(
        {
            "fold": "CROSS_FOLD_MEAN_STD",
            "accuracy": _mean([float(row["accuracy"]) for row in rows]),
            "accuracy_std": _std([float(row["accuracy"]) for row in rows]),
            "macro_f1": _mean([float(row["macro_f1"]) for row in rows]),
            "macro_f1_std": _std([float(row["macro_f1"]) for row in rows]),
            "loss": _mean([float(row["loss"]) for row in rows]),
        }
    )

    summary_path = finetune_root / "loso_finetune_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def run_loso_finetuning(args: argparse.Namespace, source_config_path: Path, base_config: dict) -> None:
    root_dir_value = args.root_dir.strip() or str((base_config.get("data", {}) or {}).get("root_dir", "")).strip()
    if not root_dir_value:
        raise ValueError("Could not resolve data.root_dir for LOSO run. Pass --root-dir explicitly or set it in config.")
    root_dir = _resolve_repo_path(root_dir_value)
    split_root = _resolve_repo_path(args.split_root.strip()) if args.split_root.strip() else (root_dir / "loso_subjectwise")
    if not split_root.exists():
        raise FileNotFoundError(f"LOSO split root not found: {split_root}")

    fold_dirs = sorted(path for path in split_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {split_root}")

    output_root_value = args.output_dir.strip() or str((base_config.get("finetune", {}) or {}).get("output_dir", "")).strip()
    if not output_root_value:
        raise ValueError("Could not resolve finetune.output_dir for LOSO run. Pass --output-dir explicitly or set it in config.")
    output_root = _resolve_repo_path(output_root_value)
    if output_root.name.startswith("fold_"):
        output_root = output_root.parent
    output_root.mkdir(parents=True, exist_ok=True)

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"[{fold_name}] finetune", flush=True)
        fold_args = argparse.Namespace(**vars(args))
        fold_args.loso = False
        fold_args.train_manifest = str((fold_dir / "manifest_train.csv").resolve())
        fold_args.val_manifest = str((fold_dir / "manifest_val.csv").resolve())
        fold_args.test_manifest = str((fold_dir / "manifest_test.csv").resolve())
        fold_args.root_dir = str(root_dir)
        fold_args.output_dir = str((output_root / fold_name).resolve())

        fold_config = apply_overrides(copy.deepcopy(base_config), fold_args)
        runtime_config_path = write_runtime_config(fold_config, source_config_path)
        try:
            run_finetuning(str(runtime_config_path))
        finally:
            runtime_config_path.unlink(missing_ok=True)

    summary_path = write_loso_summary(output_root)
    print(f"Saved summary: {summary_path}", flush=True)


def main() -> None:
    """加载配置、应用命令行覆盖，并启动下游分类微调。"""
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    if args.loso:
        run_loso_finetuning(args, config_path, config)
        return

    resolved_config = apply_overrides(config, args)
    runtime_config_path = write_runtime_config(resolved_config, config_path)
    try:
        run_finetuning(str(runtime_config_path))
    finally:
        runtime_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
