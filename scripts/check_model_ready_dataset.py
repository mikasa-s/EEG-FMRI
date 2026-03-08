from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal dataloader and model-forward check for exported EEG/fMRI data.")
    parser.add_argument("--config", default="configs/smoke_contrastive.yaml", help="Model config used to instantiate the current repository model.")
    parser.add_argument(
        "--manifest",
        default="outputs/ds002336_binary_block_model_ready/manifest_all.csv",
        help="Manifest CSV to validate.",
    )
    parser.add_argument(
        "--root-dir",
        default="outputs/ds002336_binary_block_model_ready",
        help="Dataset root directory referenced by the manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the dataloader check.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for the forward pass.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_project_modules() -> tuple[Any, Any, Any]:
    from mmcontrast.config import TrainConfig
    from mmcontrast.datasets import PairedEEGfMRIDataset
    from mmcontrast.models import EEGfMRIContrastiveModel

    return TrainConfig, PairedEEGfMRIDataset, EEGfMRIContrastiveModel


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_cfg(raw_cfg: dict) -> dict:
    cfg = {
        "train": copy.deepcopy(raw_cfg["train"]),
        "data": copy.deepcopy(raw_cfg["data"]),
        "eeg_model": copy.deepcopy(raw_cfg["eeg_model"]),
        "fmri_model": copy.deepcopy(raw_cfg["fmri_model"]),
    }
    eeg_ckpt = str(cfg["eeg_model"].get("checkpoint_path", "")).strip()
    fmri_ckpt = str(cfg["fmri_model"].get("checkpoint_path", "")).strip()
    cfg["eeg_model"]["checkpoint_path"] = str(resolve_path(eeg_ckpt)) if eeg_ckpt else ""
    cfg["fmri_model"]["gradient_csv_path"] = str(resolve_path(str(cfg["fmri_model"]["gradient_csv_path"])))
    cfg["fmri_model"]["checkpoint_path"] = str(resolve_path(fmri_ckpt)) if fmri_ckpt else ""
    return cfg


def describe_tensor(name: str, tensor: torch.Tensor) -> str:
    return f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    TrainConfig, PairedEEGfMRIDataset, EEGfMRIContrastiveModel = import_project_modules()

    config_path = resolve_path(args.config)
    manifest_path = resolve_path(args.manifest)
    root_dir = resolve_path(args.root_dir)

    raw_cfg = TrainConfig.load(str(config_path)).raw
    data_cfg = copy.deepcopy(raw_cfg.get("data", {}))
    data_cfg["manifest_csv"] = str(manifest_path)
    data_cfg["root_dir"] = str(root_dir)
    data_cfg.pop("train_manifest_csv", None)
    data_cfg.pop("val_manifest_csv", None)
    data_cfg.pop("test_manifest_csv", None)

    dataset = PairedEEGfMRIDataset(**data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    batch = next(iter(loader))
    eeg = batch["eeg"].to(device)
    fmri = batch["fmri"].to(device)
    labels = batch.get("label")

    model_cfg = build_model_cfg(raw_cfg)
    model = EEGfMRIContrastiveModel(model_cfg).to(device)
    model.eval()

    with torch.no_grad():
        out = model(eeg=eeg, fmri=fmri)

    print(f"dataset_size: {len(dataset)}")
    print(describe_tensor("eeg", eeg))
    print(describe_tensor("fmri", fmri))
    if labels is not None:
        print(f"labels: shape={tuple(labels.shape)}, values={labels.tolist()}")
    print(describe_tensor("eeg_feat", out["eeg_feat"]))
    print(describe_tensor("fmri_feat", out["fmri_feat"]))
    print(describe_tensor("eeg_embed", out["eeg_embed"]))
    print(describe_tensor("fmri_embed", out["fmri_embed"]))

    for name, tensor in out.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Model output contains non-finite values: {name}")

    print("check_passed: dataset can be loaded by the current dataloader and consumed by the current model")


if __name__ == "__main__":
    main()
