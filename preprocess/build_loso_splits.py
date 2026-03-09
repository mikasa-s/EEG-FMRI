from __future__ import annotations

"""生成留一被试交叉验证的 split。"""

import argparse
from pathlib import Path

from split_utils import write_loso_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create LOSO subject-wise folds for manifest CSV")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for generated LOSO split manifests. "
            "Recommended pattern: cache/<dataset_name>/loso_subjectwise. "
            "The script will create one subdirectory per fold under this directory."
        ),
    )
    parser.add_argument("--val-subjects", type=int, default=2, help="Number of validation subjects in each fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    write_loso_splits(manifest_path=manifest_path, output_dir=output_dir, val_subjects=int(args.val_subjects))
    print(f"Wrote LOSO folds to {output_dir}")


if __name__ == "__main__":
    main()
