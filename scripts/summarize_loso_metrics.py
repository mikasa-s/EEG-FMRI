from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize LOSO finetune metrics across folds")
    parser.add_argument("--finetune-root", required=True, help="Directory containing fold_* finetune output folders")
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path. Defaults to <finetune-root>/loso_finetune_summary.csv",
    )
    return parser.parse_args()


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else 0.0


def safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=0)) if len(series) > 1 else 0.0


def main() -> None:
    args = parse_args()
    finetune_root = Path(args.finetune_root).resolve()
    if not finetune_root.exists():
        raise FileNotFoundError(f"Finetune root not found: {finetune_root}")

    rows: list[dict[str, float | str]] = []
    for fold_dir in sorted(path for path in finetune_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics_path = fold_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "fold": fold_dir.name,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "accuracy_std": float(metrics.get("accuracy_std", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "macro_f1_std": float(metrics.get("macro_f1_std", 0.0)),
                "loss": float(metrics.get("loss", 0.0)),
            }
        )

    if not rows:
        raise RuntimeError(f"No fold test_metrics.json files found under {finetune_root}")

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    aggregate = pd.DataFrame(
        [
            {
                "fold": "CROSS_FOLD_MEAN_STD",
                "accuracy": safe_mean(df["accuracy"]),
                "accuracy_std": safe_std(df["accuracy"]),
                "macro_f1": safe_mean(df["macro_f1"]),
                "macro_f1_std": safe_std(df["macro_f1"]),
                "loss": safe_mean(df["loss"]),
            }
        ]
    )
    summary = pd.concat([df, aggregate], ignore_index=True)

    output_csv = Path(args.output_csv).resolve() if str(args.output_csv).strip() else (finetune_root / "loso_finetune_summary.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(summary.to_string(index=False))
    print("---")
    print(f"Cross-fold accuracy: {aggregate.iloc[0]['accuracy']:.4f} ± {aggregate.iloc[0]['accuracy_std']:.4f}")
    print(f"Cross-fold macro_f1: {aggregate.iloc[0]['macro_f1']:.4f} ± {aggregate.iloc[0]['macro_f1_std']:.4f}")
    print(f"Saved summary to {output_csv}")


if __name__ == "__main__":
    main()