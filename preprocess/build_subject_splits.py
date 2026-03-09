from __future__ import annotations

"""按被试划分 manifest，避免同一被试泄漏到不同 split。"""

import argparse
from pathlib import Path

from split_utils import write_subject_splits


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
    write_subject_splits(
        manifest_path=manifest_path,
        output_dir=output_dir,
        train_subjects=int(args.train_subjects),
        val_subjects=int(args.val_subjects),
        test_subjects=int(args.test_subjects),
    )
    print(f"Wrote subject-wise manifests to {output_dir}")


if __name__ == "__main__":
    main()
