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
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mmcontrast.checkpoint_utils import extract_state_dict, load_checkpoint_file
from mmcontrast.config import TrainConfig
from mmcontrast.dataset_batching import GroupedBatchSampler
from mmcontrast.datasets import PairedEEGfMRIDataset
from mmcontrast.models import EEGfMRIContrastiveModel
from mmcontrast.visualization import (
    next_indexed_output_path,
    save_confusion_matrix,
    save_cross_modal_similarity_heatmap,
    save_shared_private_tsne,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_CLASS_NAMES = {
    "ds009999": ["negative", "neutral", "positive"],
}

DATASET_CONFUSION_TITLES = {
    "ds002336": "XP1",
    "ds002338": "XP2",
    "ds009999": "SEED",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Unified visualization entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    contrastive_parser = subparsers.add_parser(
        "contrastive",
        help="Visualize shared/private embeddings from a contrastive checkpoint.",
    )
    contrastive_parser.add_argument("--config", type=str, default="configs/train_joint_contrastive.yaml")
    contrastive_parser.add_argument("--checkpoint", type=str, required=True, help="Contrastive checkpoint path.")
    contrastive_parser.add_argument("--manifest", type=str, default="", help="Optional manifest override.")
    contrastive_parser.add_argument("--root-dir", type=str, default="", help="Optional dataset root override.")
    contrastive_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/contrastive",
        help="Directory for PNG/JSON outputs.",
    )
    contrastive_parser.add_argument("--batch-size", type=int, default=32)
    contrastive_parser.add_argument("--num-workers", type=int, default=0)
    contrastive_parser.add_argument("--max-samples", type=int, default=0)
    contrastive_parser.add_argument("--sample-seed", type=int, default=None, help="Optional seed for random batch-aligned slice selection when --max-samples is used.")
    contrastive_parser.add_argument("--tsne-max-points", type=int, default=200)
    contrastive_parser.add_argument("--heatmap-max-points", type=int, default=48)
    contrastive_parser.add_argument("--device", type=str, default="", help="Explicit device, e.g. cpu or cuda:0.")
    contrastive_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Generic override in the form section.key=value. Can be repeated.",
    )

    offline_parser = subparsers.add_parser(
        "offline-loso",
        help="Evaluate existing LOSO finetune checkpoints and generate LOSO confusion matrices.",
    )
    offline_parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name used in output file names.")
    offline_parser.add_argument("--config", type=str, required=True, help="Finetune config path.")
    offline_parser.add_argument(
        "--checkpoints-root",
        type=str,
        required=True,
        help="Root directory that contains fold_*/checkpoints/best.pth.",
    )
    offline_parser.add_argument("--checkpoint-relpath", type=str, default="checkpoints/best.pth")
    offline_parser.add_argument("--split-root", type=str, default="")
    offline_parser.add_argument("--root-dir", type=str, default="")
    offline_parser.add_argument("--output-dir", type=str, default="")
    offline_parser.add_argument("--eval-batch-size", type=int, default=None)
    offline_parser.add_argument("--num-workers", type=int, default=None)
    offline_parser.add_argument("--force-cpu", action="store_true")
    offline_parser.add_argument(
        "--allow-missing-pretrain-checkpoint",
        action="store_true",
        help="If the pretrain checkpoint inferred by finetune config is missing, fall back to random initialization.",
    )
    offline_parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Optional comma-separated class names for confusion matrix axes.",
    )
    return parser


def assign_nested_value(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = payload
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if args.manifest.strip():
        assign_nested_value(config, "data.manifest_csv", args.manifest.strip())
    if args.root_dir.strip():
        assign_nested_value(config, "data.root_dir", args.root_dir.strip())
    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected section.key=value")
        dotted_key, raw_value = override.split("=", 1)
        assign_nested_value(config, dotted_key.strip(), yaml.safe_load(raw_value))
    return config


def build_dataset(cfg: dict[str, Any]) -> PairedEEGfMRIDataset:
    data_cfg = dict(cfg["data"])
    manifest_path = str(data_cfg.get("manifest_csv", "")).strip()
    if not manifest_path:
        raise ValueError("data.manifest_csv must be configured for visualization")
    data_cfg["manifest_csv"] = str(resolve_path(manifest_path))
    if data_cfg.get("root_dir"):
        data_cfg["root_dir"] = str(resolve_path(str(data_cfg["root_dir"])))
    data_cfg.pop("train_manifest_csv", None)
    data_cfg.pop("val_manifest_csv", None)
    data_cfg.pop("test_manifest_csv", None)
    data_cfg.pop("expected_eeg_shape", None)
    data_cfg.pop("expected_fmri_shape", None)
    data_cfg["require_eeg"] = True
    data_cfg["require_fmri"] = True
    data_cfg["require_band_power"] = False
    return PairedEEGfMRIDataset(**data_cfg)


def build_model(cfg: dict[str, Any], device: torch.device) -> EEGfMRIContrastiveModel:
    model_cfg = {
        "train": dict(cfg["train"]),
        "data": dict(cfg["data"]),
        "eeg_model": dict(cfg["eeg_model"]),
        "fmri_model": dict(cfg["fmri_model"]),
    }
    model_cfg["eeg_model"]["checkpoint_path"] = ""
    model_cfg["fmri_model"]["checkpoint_path"] = ""
    model = EEGfMRIContrastiveModel(model_cfg).to(device)
    return model


def load_model_checkpoint(model: EEGfMRIContrastiveModel, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint_file(str(checkpoint_path))
    state_dict = extract_state_dict(checkpoint, preferred_keys=("model", "module", "state_dict"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_count": int(len(missing)),
        "unexpected_count": int(len(unexpected)),
        "missing": [str(item) for item in missing[:20]],
        "unexpected": [str(item) for item in unexpected[:20]],
        "device": str(device),
    }


def collect_embeddings(
    model: EEGfMRIContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {
        "eeg_shared": [],
        "eeg_private": [],
        "fmri_shared": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            fmri = batch["fmri"].to(device, non_blocking=True)
            batch_out = model(eeg=eeg, fmri=fmri)
            outputs["eeg_shared"].append(batch_out["eeg_shared"].detach().cpu())
            outputs["eeg_private"].append(batch_out["eeg_private"].detach().cpu())
            outputs["fmri_shared"].append(batch_out["fmri_shared"].detach().cpu())
    collected = {key: torch.cat(value, dim=0) for key, value in outputs.items() if value}
    if not collected:
        raise RuntimeError("No embeddings were collected for visualization.")
    return collected


def run_contrastive_visualization(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    TrainConfig(config).validate(base_dir=str(PROJECT_ROOT))

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_path(args.checkpoint)
    requested_device = args.device.strip()
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(config)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        max_samples = int(args.max_samples)
        batch_size = max(1, int(args.batch_size))
        max_start = len(dataset) - max_samples
        candidate_starts = list(range(0, max_start + 1, batch_size))
        if not candidate_starts:
            candidate_starts = [0]
        if args.sample_seed is None:
            start_pos = int(torch.randint(0, len(candidate_starts), (1,)).item())
        else:
            generator = torch.Generator()
            generator.manual_seed(int(args.sample_seed))
            start_pos = int(torch.randint(0, len(candidate_starts), (1,), generator=generator).item())
        start_index = int(candidate_starts[start_pos])
        sample_indices = list(range(start_index, start_index + max_samples))
        dataset = Subset(dataset, sample_indices)
    loader = DataLoader(
        dataset,
        batch_sampler=GroupedBatchSampler(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            group_field="dataset",
            shuffle=False,
            drop_last=False,
            world_size=1,
            rank=0,
            seed=int(args.sample_seed) if args.sample_seed is not None else 42,
        ),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers > 0),
    )

    model = build_model(config, device=device)
    checkpoint_report = load_model_checkpoint(model, checkpoint_path, device=device)
    embeddings = collect_embeddings(model, loader, device=device)
    tsne_path = next_indexed_output_path(output_dir, "tsne_shared_private", ".svg")
    heatmap_path = next_indexed_output_path(output_dir, "cross_modal_similarity_heatmap", ".svg")
    summary_path = next_indexed_output_path(output_dir, "visualization_summary", ".json")
    tsne_report = save_shared_private_tsne(
        embeddings["eeg_shared"],
        embeddings["eeg_private"],
        embeddings["fmri_shared"],
        output_path=tsne_path,
        max_points=max(3, int(args.tsne_max_points)),
    )
    heatmap_report = save_cross_modal_similarity_heatmap(
        embeddings["eeg_shared"],
        embeddings["fmri_shared"],
        output_path=heatmap_path,
        max_points=max(2, int(args.heatmap_max_points)),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_samples": int(embeddings["eeg_shared"].shape[0]),
        "checkpoint_report": checkpoint_report,
        "tsne_report": tsne_report,
        "heatmap_report": heatmap_report,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


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


def run_offline_loso_visualization(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    data_cfg = dict(cfg.get("data", {}) or {})
    root_dir_value = args.root_dir.strip() or str(data_cfg.get("root_dir", "")).strip()
    if not root_dir_value:
        raise ValueError("Could not resolve data.root_dir from config. Pass --root-dir explicitly.")
    root_dir = Path(root_dir_value)
    if not root_dir.is_absolute():
        root_dir = (PROJECT_ROOT / root_dir).resolve()

    split_root = Path(args.split_root).resolve() if args.split_root.strip() else (root_dir / "loso_subjectwise").resolve()
    checkpoints_root = Path(args.checkpoints_root)
    if not checkpoints_root.is_absolute():
        checkpoints_root = (PROJECT_ROOT / checkpoints_root).resolve()

    output_dir = Path(args.output_dir).resolve() if args.output_dir.strip() else (PROJECT_ROOT / "outputs" / f"{args.dataset_name}_offline_eval").resolve()
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
            str(PROJECT_ROOT / "run_finetune.py"),
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
        if args.allow_missing_pretrain_checkpoint:
            command.append("--allow-missing-pretrain-checkpoint")

        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Offline evaluation failed for {fold_name} with exit code {completed.returncode}")

        metrics_path = fold_output_dir / "test_metrics.json"
        logits_path = fold_output_dir / "test_logits.csv"
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
            output_path=output_dir / f"confusion_matrix_{args.dataset_name}_loso.svg",
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "contrastive":
        run_contrastive_visualization(args)
        return
    if args.command == "offline-loso":
        run_offline_loso_visualization(args)
        return
    parser.error(f"Unknown visualization command: {args.command}")


if __name__ == "__main__":
    main()
