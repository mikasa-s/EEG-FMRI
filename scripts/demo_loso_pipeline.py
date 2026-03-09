from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


if str(PROJECT_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmcontrast.datasets.paired_manifest_dataset import PairedEEGfMRIDataset
from mmcontrast.losses import SymmetricInfoNCELoss
from mmcontrast.models.multimodal_model import EEGfMRIContrastiveModel
from mmcontrast.runner import run_training
from mmcontrast.finetune_runner import run_finetuning


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo_loso_pipeline"
BASE_CONFIG = PROJECT_ROOT / "configs" / "train_contrastive_true450_5patch.yaml"
BASE_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "finetune_classifier_true450_5patch.yaml"


def build_dummy_dataset(output_root: Path) -> Path:
    eeg_dir = output_root / "eeg"
    fmri_dir = output_root / "fmri"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    fmri_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    subjects = ["sub-a", "sub-b", "sub-c", "sub-d", "sub-e"]
    for index, subject in enumerate(subjects):
        sample_id = f"{subject}_sample-01"
        eeg = np.full((63, 20, 200), fill_value=index + 1, dtype=np.float32)
        fmri = np.full((450, 10), fill_value=(index + 1) * 10, dtype=np.float32)

        np.save(eeg_dir / f"{sample_id}.npy", eeg)
        np.save(fmri_dir / f"{sample_id}.npy", fmri)

        rows.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "task": "dummy",
                "trial_type": "dummy",
                "eeg_path": f"eeg/{sample_id}.npy",
                "fmri_path": f"fmri/{sample_id}.npy",
                "label": index % 2,
                "label_name": "dummy",
                "eeg_shape": "63x20x200",
                "fmri_shape": "450x10",
            }
        )

    manifest_path = output_root / "manifest_all.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def run_loso(manifest_path: Path, output_root: Path) -> Path:
    loso_dir = output_root / "loso_subjectwise"
    script_path = PROJECT_ROOT / "scripts" / "build_loso_splits.py"
    command = [
        sys.executable,
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(loso_dir),
        "--val-subjects",
        "1",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return loso_dir


def print_summary(loso_dir: Path) -> None:
    summary = pd.read_csv(loso_dir / "loso_summary.csv")
    print("LOSO summary:")
    print(summary.to_string(index=False))
    print("---")

    for fold_dir in sorted(path for path in loso_dir.iterdir() if path.is_dir()):
        print(fold_dir.name)
        for split in ["train", "val", "test"]:
            split_df = pd.read_csv(fold_dir / f"manifest_{split}.csv")
            subjects = split_df["subject"].tolist()
            samples = split_df["sample_id"].tolist()
            print(f"  {split}: rows={len(split_df)}, subjects={subjects}, samples={samples}")
        print("---")


def load_base_model_config() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["train"]["use_amp"] = False
    cfg["train"]["force_cpu"] = True
    gradient_path = Path(str(cfg["fmri_model"]["gradient_csv_path"]))
    if not gradient_path.is_absolute():
        cfg["fmri_model"]["gradient_csv_path"] = str((PROJECT_ROOT / gradient_path).resolve())
    checkpoint_path = str(cfg["fmri_model"].get("checkpoint_path", "")).strip()
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_absolute():
            cfg["fmri_model"]["checkpoint_path"] = str((PROJECT_ROOT / checkpoint_file).resolve())
    eeg_checkpoint = str(cfg["eeg_model"].get("checkpoint_path", "")).strip()
    if eeg_checkpoint:
        checkpoint_file = Path(eeg_checkpoint)
        if not checkpoint_file.is_absolute():
            cfg["eeg_model"]["checkpoint_path"] = str((PROJECT_ROOT / checkpoint_file).resolve())
    return cfg


def load_base_finetune_config() -> dict:
    with open(BASE_FINETUNE_CONFIG, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["train"]["force_cpu"] = True
    cfg["finetune"]["use_amp"] = False
    cfg["finetune"]["selection_metric"] = "accuracy"
    gradient_path = Path(str(cfg["fmri_model"]["gradient_csv_path"]))
    if not gradient_path.is_absolute():
        cfg["fmri_model"]["gradient_csv_path"] = str((PROJECT_ROOT / gradient_path).resolve())
    return cfg


def load_split_batch(manifest_path: Path, root_dir: Path) -> dict[str, torch.Tensor | list[str] | int]:
    dataset = PairedEEGfMRIDataset(
        manifest_csv=str(manifest_path),
        root_dir=str(root_dir),
        normalize_eeg=True,
        normalize_fmri=True,
    )
    sample = dataset[0]
    return {
        "sample_id": sample["sample_id"],
        "label": sample.get("label"),
        "eeg": sample["eeg"].unsqueeze(0),
        "fmri": sample["fmri"].unsqueeze(0),
    }


def run_model_check(loso_dir: Path, data_root: Path) -> None:
    cfg = load_base_model_config()
    model = EEGfMRIContrastiveModel(cfg).eval()
    criterion = SymmetricInfoNCELoss(float(cfg["train"].get("temperature", 0.07)))

    fold_dir = loso_dir / "fold_sub-a"
    print(f"Model check fold: {fold_dir.name}")
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            manifest_path = fold_dir / f"manifest_{split}.csv"
            batch = load_split_batch(manifest_path, root_dir=data_root)
            eeg = batch["eeg"]
            fmri = batch["fmri"]
            outputs = model(eeg=eeg, fmri=fmri)
            loss = criterion(outputs["eeg_embed"], outputs["fmri_embed"])
            print(
                f"  {split}: sample_id={batch['sample_id']}, label={batch['label']}, "
                f"eeg={tuple(eeg.shape)}, fmri={tuple(fmri.shape)}, "
                f"eeg_feat={tuple(outputs['eeg_feat'].shape)}, fmri_feat={tuple(outputs['fmri_feat'].shape)}, "
                f"eeg_embed={tuple(outputs['eeg_embed'].shape)}, fmri_embed={tuple(outputs['fmri_embed'].shape)}, "
                f"loss={float(loss.item()):.6f}"
            )


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def build_demo_train_config(loso_dir: Path, data_root: Path, fold_name: str) -> Path:
    cfg = load_base_model_config()
    fold_dir = loso_dir / fold_name
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 1
    cfg["train"]["eval_batch_size"] = 1
    cfg["train"]["num_workers"] = 0
    cfg["train"]["log_interval"] = 1
    cfg["train"]["eval_interval"] = 1
    cfg["train"]["output_dir"] = str((data_root / "contrastive_demo_run" / fold_name).resolve())
    cfg["data"]["train_manifest_csv"] = str((fold_dir / "manifest_train.csv").resolve())
    cfg["data"]["val_manifest_csv"] = str((fold_dir / "manifest_val.csv").resolve())
    cfg["data"]["root_dir"] = str(data_root.resolve())
    cfg["data"].pop("test_manifest_csv", None)
    config_path = data_root / f"train_{fold_name}.yaml"
    write_yaml(config_path, cfg)
    return config_path


def build_demo_finetune_config(loso_dir: Path, data_root: Path, fold_name: str, contrastive_ckpt: Path) -> Path:
    cfg = load_base_finetune_config()
    fold_dir = loso_dir / fold_name
    cfg["train"]["batch_size"] = 1
    cfg["train"]["eval_batch_size"] = 1
    cfg["train"]["num_workers"] = 0
    cfg["data"]["train_manifest_csv"] = str((fold_dir / "manifest_train.csv").resolve())
    cfg["data"]["val_manifest_csv"] = str((fold_dir / "manifest_val.csv").resolve())
    cfg["data"]["test_manifest_csv"] = str((fold_dir / "manifest_test.csv").resolve())
    cfg["data"]["root_dir"] = str(data_root.resolve())
    cfg["finetune"]["epochs"] = 1
    cfg["finetune"]["log_interval"] = 1
    cfg["finetune"]["eval_interval"] = 1
    cfg["finetune"]["output_dir"] = str((data_root / "finetune_demo_run" / fold_name).resolve())
    cfg["finetune"]["contrastive_checkpoint_path"] = str(contrastive_ckpt.resolve())
    config_path = data_root / f"finetune_{fold_name}.yaml"
    write_yaml(config_path, cfg)
    return config_path


def run_full_demo_training(loso_dir: Path, data_root: Path) -> None:
    fold_name = "fold_sub-a"
    train_config = build_demo_train_config(loso_dir, data_root, fold_name)
    print(f"Running contrastive demo with config: {train_config}")
    run_training(str(train_config))

    contrastive_ckpt = data_root / "contrastive_demo_run" / fold_name / "checkpoints" / "best.pth"
    finetune_config = build_demo_finetune_config(loso_dir, data_root, fold_name, contrastive_ckpt)
    print(f"Running finetune demo with config: {finetune_config}")
    run_finetuning(str(finetune_config))

    test_metrics_path = data_root / "finetune_demo_run" / fold_name / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, "r", encoding="utf-8") as handle:
            print(f"Finetune test metrics: {handle.read()}")

    summary_script = PROJECT_ROOT / "scripts" / "summarize_loso_metrics.py"
    summary_command = [
        sys.executable,
        str(summary_script),
        "--finetune-root",
        str((data_root / "finetune_demo_run").resolve()),
    ]
    subprocess.run(summary_command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = build_dummy_dataset(OUTPUT_ROOT)
    loso_dir = run_loso(manifest_path, OUTPUT_ROOT)
    print(f"Dummy manifest: {manifest_path}")
    print(f"LOSO output dir: {loso_dir}")
    print_summary(loso_dir)
    run_model_check(loso_dir, data_root=OUTPUT_ROOT)
    run_full_demo_training(loso_dir, data_root=OUTPUT_ROOT)


if __name__ == "__main__":
    main()