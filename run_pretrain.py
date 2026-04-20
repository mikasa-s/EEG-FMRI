from __future__ import annotations

"""对比学习训练入口脚本。"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from mmcontrast.contrastive_runner import run_contrastive_training
from mmcontrast.pretrain_pathing import (
    FULL_MODE,
    STRICT_MODE,
    create_strict_manifest,
    infer_pretrain_objective_name,
    list_dataset_subjects_from_manifest,
    resolve_pretrain_cache_dir,
    resolve_pretrain_output_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# 允许直接从项目根目录执行脚本，而不依赖外部 PYTHONPATH 配置。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，支持直接覆盖常用训练配置。"""
    parser = argparse.ArgumentParser("EEG-fMRI contrastive training")
    parser.add_argument("--config", type=str, default="configs/train_joint_contrastive.yaml")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--pretrain-mode", type=str, choices=[FULL_MODE, STRICT_MODE], default="")
    parser.add_argument("--target-dataset", type=str, default="", help="Required with --pretrain-mode strict.")
    parser.add_argument("--held-out-subject", type=str, default="", help="Required with --pretrain-mode strict.")
    parser.add_argument("--pretrain-cache-root", type=str, default="", help="Optional cache root override. Defaults to cache/joint_contrastive.")
    parser.add_argument("--pretrain-output-root", type=str, default="", help="Optional output root override for --pretrain-mode.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume-path", type=str, default="")
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
    pretrain_mode = str(args.pretrain_mode).strip().lower()
    if pretrain_mode:
        objective_name = infer_pretrain_objective_name(args.config, config)
        resolved_cache_dir = resolve_pretrain_cache_dir(
            project_root=PROJECT_ROOT,
            cache_root=args.pretrain_cache_root.strip(),
        )
        resolved_output_dir = resolve_pretrain_output_dir(
            project_root=PROJECT_ROOT,
            mode=pretrain_mode,
            objective_name=objective_name,
            target_dataset=args.target_dataset.strip(),
            held_out_subject=args.held_out_subject.strip(),
            output_root=args.pretrain_output_root.strip(),
        )
        if not args.manifest.strip():
            args.manifest = str((resolved_cache_dir / "manifest_all.csv").resolve())
        if not args.root_dir.strip():
            args.root_dir = str(resolved_cache_dir)
        if not args.output_dir.strip():
            args.output_dir = str(resolved_output_dir)

    mapping: list[tuple[str, object]] = [
        ("data.manifest_csv", args.manifest.strip()),
        ("data.root_dir", args.root_dir.strip()),
        ("train.output_dir", args.output_dir.strip()),
        ("train.epochs", args.epochs),
        ("train.batch_size", args.batch_size),
        ("train.num_workers", args.num_workers),
        ("train.lr", args.lr),
        ("train.resume_path", args.resume_path.strip()),
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


def _run_one_pretrain(config_path: Path, base_config: dict, args: argparse.Namespace) -> None:
    strict_manifest_path: Path | None = None
    strict_manifest_info_path: Path | None = None
    pretrain_mode = str(args.pretrain_mode).strip().lower()
    if pretrain_mode == STRICT_MODE:
        if not args.target_dataset.strip():
            raise ValueError("--target-dataset is required when --pretrain-mode strict is used.")
        if not args.held_out_subject.strip():
            raise ValueError("--held-out-subject is required when running one strict pretrain job.")
        runtime_dir = PROJECT_ROOT / ".runtime_configs"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        base_manifest = Path(args.manifest).resolve() if args.manifest.strip() else (
            resolve_pretrain_cache_dir(project_root=PROJECT_ROOT, cache_root=args.pretrain_cache_root.strip()) / "manifest_all.csv"
        ).resolve()
        safe_subject_name = args.held_out_subject.strip().replace("/", "_").replace("\\", "_")
        strict_manifest_path = runtime_dir / (
            f"strict_manifest_{args.target_dataset.strip()}_{safe_subject_name}.csv"
        )
        report = create_strict_manifest(
            source_manifest_path=base_manifest,
            output_manifest_path=strict_manifest_path,
            target_dataset=args.target_dataset.strip(),
            held_out_subject=args.held_out_subject.strip(),
        )
        strict_manifest_info_path = strict_manifest_path.with_suffix(".json")
        with strict_manifest_info_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        args.manifest = str(strict_manifest_path)

    resolved_config = apply_overrides(base_config, args)
    runtime_config_path = write_runtime_config(resolved_config, config_path)
    try:
        run_contrastive_training(str(runtime_config_path))
    finally:
        runtime_config_path.unlink(missing_ok=True)
        if strict_manifest_path is not None:
            strict_manifest_path.unlink(missing_ok=True)
        if strict_manifest_info_path is not None:
            strict_manifest_info_path.unlink(missing_ok=True)


def main() -> None:
    """加载配置、应用命令行覆盖，并启动对比学习训练流程。"""
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    pretrain_mode = str(args.pretrain_mode).strip().lower()
    if pretrain_mode == STRICT_MODE:
        if not args.target_dataset.strip():
            raise ValueError("--target-dataset is required when --pretrain-mode strict is used.")
        if not args.held_out_subject.strip():
            base_manifest = Path(args.manifest).resolve() if args.manifest.strip() else (
                resolve_pretrain_cache_dir(project_root=PROJECT_ROOT, cache_root=args.pretrain_cache_root.strip()) / "manifest_all.csv"
            ).resolve()
            held_out_subjects = list_dataset_subjects_from_manifest(base_manifest, args.target_dataset.strip())
            for held_out_subject in held_out_subjects:
                subject_args = argparse.Namespace(**vars(args))
                subject_args.held_out_subject = held_out_subject
                print(
                    f"Running strict pretrain for dataset={subject_args.target_dataset.strip()}, "
                    f"held_out_subject={held_out_subject}",
                    flush=True,
                )
                _run_one_pretrain(config_path, dict(config), subject_args)
            return

    _run_one_pretrain(config_path, config, args)
if __name__ == "__main__":
    main()
