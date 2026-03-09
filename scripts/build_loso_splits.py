from __future__ import annotations

"""生成留一被试交叉验证的 split 与对应配置文件。"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create LOSO subject-wise folds for manifest CSV")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for generated LOSO split manifests. "
            "Recommended pattern: outputs/<dataset_name>/loso_subjectwise. "
            "The script will create one subdirectory per fold under this directory."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default="",
        help=(
            "Optional directory for generated per-fold YAML configs. "
            "Leave empty to only generate manifest files."
        ),
    )
    parser.add_argument(
        "--write-fold-configs",
        action="store_true",
        help="Also write per-fold train/finetune YAML files. Disabled by default because these configs are derived artifacts.",
    )
    parser.add_argument(
        "--contrastive-template",
        default="configs/train_contrastive_binary_block.yaml",
        help="Base contrastive config template",
    )
    parser.add_argument(
        "--finetune-template",
        default="configs/finetune_classifier_binary_block.yaml",
        help="Base finetune config template",
    )
    parser.add_argument("--val-subjects", type=int, default=2, help="Number of validation subjects in each fold")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def format_label_distribution(split_df: pd.DataFrame) -> dict[int, int]:
    if "label" not in split_df.columns:
        return {}
    return {int(key): int(value) for key, value in split_df["label"].value_counts().sort_index().items()}


def choose_val_subjects(subjects: list[str], held_out_index: int, val_subjects: int) -> list[str]:
    rotated = subjects[held_out_index + 1 :] + subjects[:held_out_index]
    return rotated[:val_subjects]


def update_fold_configs(
    contrastive_template: dict[str, Any],
    finetune_template: dict[str, Any],
    fold_name: str,
    fold_rel_dir: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contrastive_cfg = yaml.safe_load(yaml.safe_dump(contrastive_template, sort_keys=False, allow_unicode=True))
    finetune_cfg = yaml.safe_load(yaml.safe_dump(finetune_template, sort_keys=False, allow_unicode=True))

    for cfg in (contrastive_cfg, finetune_cfg):
        cfg["data"]["train_manifest_csv"] = f"{fold_rel_dir}/manifest_train.csv"
        cfg["data"]["val_manifest_csv"] = f"{fold_rel_dir}/manifest_val.csv"
        cfg["data"]["test_manifest_csv"] = f"{fold_rel_dir}/manifest_test.csv"

    contrastive_cfg["train"]["output_dir"] = f"outputs/contrastive_binary_block_loso/{fold_name}"
    finetune_cfg["finetune"]["output_dir"] = f"outputs/finetune_binary_block_loso/{fold_name}"
    finetune_cfg["finetune"]["contrastive_checkpoint_path"] = ""
    return contrastive_cfg, finetune_cfg


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_fold_configs = bool(args.write_fold_configs)
    config_dir: Path | None = None
    if write_fold_configs:
        if not str(args.config_dir).strip():
            raise ValueError("--write-fold-configs requires --config-dir")
        config_dir = Path(args.config_dir).resolve()
        config_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    if "subject" not in manifest.columns:
        raise ValueError("Manifest must contain a 'subject' column for LOSO splitting")

    subjects = sorted(manifest["subject"].dropna().unique().tolist())
    if args.val_subjects <= 0:
        raise ValueError("--val-subjects must be positive")
    if args.val_subjects >= len(subjects):
        raise ValueError("--val-subjects must be smaller than the total number of subjects")

    contrastive_template: dict[str, Any] | None = None
    finetune_template: dict[str, Any] | None = None
    if write_fold_configs:
        contrastive_template = load_yaml(Path(args.contrastive_template).resolve())
        finetune_template = load_yaml(Path(args.finetune_template).resolve())

    summary_rows: list[dict[str, object]] = []
    for held_out_index, test_subject in enumerate(subjects):
        val_subjects = choose_val_subjects(subjects, held_out_index, args.val_subjects)
        train_subjects = [subject for subject in subjects if subject not in {test_subject, *val_subjects}]

        fold_name = f"fold_{test_subject}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        split_specs = {
            "train": train_subjects,
            "val": val_subjects,
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

        if write_fold_configs:
            fold_rel_dir = Path("outputs") / output_dir.relative_to(manifest_path.parent.parent) / fold_name
            contrastive_cfg, finetune_cfg = update_fold_configs(
                contrastive_template=contrastive_template,
                finetune_template=finetune_template,
                fold_name=fold_name,
                fold_rel_dir=fold_rel_dir.as_posix(),
            )
            dump_yaml(config_dir / f"train_contrastive_{fold_name}.yaml", contrastive_cfg)
            dump_yaml(config_dir / f"finetune_classifier_{fold_name}.yaml", finetune_cfg)

    pd.DataFrame(summary_rows).to_csv(output_dir / "loso_summary.csv", index=False)
    print(f"Wrote LOSO folds to {output_dir}")
    if write_fold_configs and config_dir is not None:
        print(f"Wrote fold configs to {config_dir}")
    else:
        print("Skipped per-fold YAML generation. Use --write-fold-configs if you explicitly need derived fold configs.")


if __name__ == "__main__":
    main()