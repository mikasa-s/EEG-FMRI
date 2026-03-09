from __future__ import annotations

"""按被试划分 manifest，避免同一被试泄漏到不同 split。"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create subject-wise splits for manifest CSV")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV path")
    parser.add_argument("--output-dir", required=True, help="Directory for split manifests")
    parser.add_argument("--train-subjects", type=int, default=6, help="Number of subjects in train split")
    parser.add_argument("--val-subjects", type=int, default=2, help="Number of subjects in val split")
    parser.add_argument("--test-subjects", type=int, default=2, help="Number of subjects in test split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    if "subject" not in manifest.columns:
        raise ValueError("Manifest must contain a 'subject' column for subject-wise splitting")

    subjects = sorted(manifest["subject"].dropna().unique().tolist())
    requested = args.train_subjects + args.val_subjects + args.test_subjects
    if requested > len(subjects):
        raise ValueError(
            f"Requested {requested} subjects across splits, but manifest only contains {len(subjects)} subjects"
        )

    train_subjects = subjects[: args.train_subjects]
    val_subjects = subjects[args.train_subjects : args.train_subjects + args.val_subjects]
    test_subjects = subjects[
        args.train_subjects + args.val_subjects : args.train_subjects + args.val_subjects + args.test_subjects
    ]

    split_map = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    summary_rows: list[dict[str, object]] = []
    for split_name, split_subjects in split_map.items():
        split_df = manifest[manifest["subject"].isin(split_subjects)].copy()
        split_df.to_csv(output_dir / f"manifest_{split_name}.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "subjects": ",".join(split_subjects),
                "num_subjects": len(split_subjects),
                "num_samples": int(len(split_df)),
                "label_distribution": split_df["label"].value_counts().sort_index().to_dict() if "label" in split_df.columns else {},
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False)
    print(f"Wrote subject-wise manifests to {output_dir}")


if __name__ == "__main__":
    main()