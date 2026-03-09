from __future__ import annotations

"""分类微调入口脚本。"""

import argparse
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
    parser.add_argument("--config", type=str, default="configs/finetune_classifier_binary_block.yaml")
    parser.add_argument("--train-manifest", type=str, default="")
    parser.add_argument("--val-manifest", type=str, default="")
    parser.add_argument("--test-manifest", type=str, default="")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--contrastive-checkpoint", type=str, default="")
    parser.add_argument("--selection-metric", type=str, default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    mapping: list[tuple[str, object]] = [
        ("data.train_manifest_csv", args.train_manifest.strip()),
        ("data.val_manifest_csv", args.val_manifest.strip()),
        ("data.test_manifest_csv", args.test_manifest.strip()),
        ("data.root_dir", args.root_dir.strip()),
        ("finetune.output_dir", args.output_dir.strip()),
        ("finetune.contrastive_checkpoint_path", args.contrastive_checkpoint.strip()),
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

    if args.force_cpu:
        assign_nested_value(config, "train.force_cpu", True)

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


def main() -> None:
    """加载配置、应用命令行覆盖，并启动下游分类微调。"""
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    resolved_config = apply_overrides(config, args)
    runtime_config_path = write_runtime_config(resolved_config, config_path)
    run_finetuning(str(runtime_config_path))


if __name__ == "__main__":
    main()
