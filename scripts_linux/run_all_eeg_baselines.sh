#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="configs/finetune_ds002336.yaml"
MODELS="svm,labram,cbramod,eeg_deformer,eegnet,conformer,tsception"
OUTPUT_ROOT=""
EPOCHS=""
BATCH_SIZE=""
EVAL_BATCH_SIZE=""
NUM_WORKERS=""
FORCE_CPU="false"
PYTHON_EXE=""
EXTRA_SET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        --set) EXTRA_SET_ARGS+=("$2"); shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${PYTHON_EXE}" ]]; then
    PYTHON="${PYTHON_EXE}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="python"
fi

cd "${REPO_ROOT}"

read_config_value() {
    local config_path="$1"
    local key="$2"
    "${PYTHON}" - <<'PY' "${config_path}" "${key}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
dotted_key = sys.argv[2]
with config_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
cursor = payload
for part in dotted_key.split("."):
    if not isinstance(cursor, dict):
        cursor = ""
        break
    cursor = cursor.get(part, "")
if cursor is None:
    cursor = ""
print(cursor)
PY
}

write_model_summary() {
    local model_root="$1"
    "${PYTHON}" - <<'PY' "${model_root}"
import csv
import json
import os
import statistics
import sys

root = sys.argv[1]
rows = []
for name in sorted(os.listdir(root)):
    if not name.startswith("fold_"):
        continue
    metrics_path = os.path.join(root, name, "final_metrics.json")
    if not os.path.isfile(metrics_path):
        metrics_path = os.path.join(root, name, "test_metrics.json")
    if not os.path.isfile(metrics_path):
        continue
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle) or {}
    test_metrics = metrics.get("test_metrics") if isinstance(metrics, dict) else None
    source = test_metrics if isinstance(test_metrics, dict) else metrics
    rows.append({
        "fold": name,
        "accuracy": float(source.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(source.get("macro_f1", 0.0) or 0.0),
        "loss": float(source.get("loss", 0.0) or 0.0),
        "best_score": float(metrics.get("best_score", 0.0) or 0.0) if isinstance(metrics, dict) else 0.0,
        "best_epoch": int(metrics.get("best_epoch", 0) or 0) if isinstance(metrics, dict) else 0,
    })

if not rows:
    raise SystemExit(f"No metrics found under {root}")

def mean(vals):
    return statistics.mean(vals) if vals else 0.0

def std(vals):
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0

summary = {
    "fold": "MEAN_STD",
    "accuracy": mean([r["accuracy"] for r in rows]),
    "macro_f1": mean([r["macro_f1"] for r in rows]),
    "loss": mean([r["loss"] for r in rows]),
    "best_score": mean([r["best_score"] for r in rows]),
    "best_epoch": mean([r["best_epoch"] for r in rows]),
    "accuracy_std": std([r["accuracy"] for r in rows]),
    "macro_f1_std": std([r["macro_f1"] for r in rows]),
    "loss_std": std([r["loss"] for r in rows]),
}

for row in rows:
    row["accuracy_std"] = ""
    row["macro_f1_std"] = ""
    row["loss_std"] = ""
rows.append(summary)

output_csv = os.path.join(root, "baseline_summary.csv")
with open(output_csv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["fold", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "loss", "loss_std", "best_score", "best_epoch"],
    )
    writer.writeheader()
    writer.writerows(rows)
print(output_csv)
PY
}

write_global_summary() {
    local output_root="$1"
    "${PYTHON}" - <<'PY' "${output_root}"
import csv
import os
import sys

root = sys.argv[1]
rows = []
for model_name in sorted(os.listdir(root)):
    model_dir = os.path.join(root, model_name)
    if not os.path.isdir(model_dir):
        continue
    summary_path = os.path.join(model_dir, "baseline_summary.csv")
    if not os.path.isfile(summary_path):
        continue
    with open(summary_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("fold") == "MEAN_STD":
                rows.append({
                    "model": model_name,
                    "accuracy_mean": row.get("accuracy", ""),
                    "accuracy_std": row.get("accuracy_std", ""),
                    "macro_f1_mean": row.get("macro_f1", ""),
                    "macro_f1_std": row.get("macro_f1_std", ""),
                    "loss_mean": row.get("loss", ""),
                    "loss_std": row.get("loss_std", ""),
                    "best_score_mean": row.get("best_score", ""),
                    "best_epoch_mean": row.get("best_epoch", ""),
                })
                break

if not rows:
    raise SystemExit(f"No per-model summaries found under {root}")

output_csv = os.path.join(root, "all_baselines_summary.csv")
with open(output_csv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "loss_mean",
            "loss_std",
            "best_score_mean",
            "best_epoch_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
print(output_csv)
PY
}

CONFIG_PATH="$("${PYTHON}" - <<'PY' "${CONFIG_PATH}"
import os, sys
print(os.path.normpath(sys.argv[1]))
PY
)"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

ROOT_DIR="$(read_config_value "${CONFIG_PATH}" "data.root_dir")"
if [[ -z "${ROOT_DIR}" ]]; then
    echo "data.root_dir is empty in ${CONFIG_PATH}" >&2
    exit 1
fi

LOSO_DIR="${ROOT_DIR}/loso_subjectwise"
if [[ ! -d "${LOSO_DIR}" ]]; then
    echo "LOSO directory not found: ${LOSO_DIR}" >&2
    exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    CONFIG_STEM="$(basename "${CONFIG_PATH}" .yaml)"
    OUTPUT_ROOT="outputs/${CONFIG_STEM}_all_baselines"
fi
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
shopt -s nullglob
FOLD_DIRS=("${LOSO_DIR}"/fold_*)
shopt -u nullglob

if [[ ${#FOLD_DIRS[@]} -eq 0 ]]; then
    echo "No fold directories found under ${LOSO_DIR}" >&2
    exit 1
fi

for raw_model in "${MODEL_LIST[@]}"; do
    model="$(echo "${raw_model}" | xargs)"
    if [[ -z "${model}" ]]; then
        continue
    fi
    model_output_root="${OUTPUT_ROOT}/${model}"
    mkdir -p "${model_output_root}"
    echo "[baseline=${model}]"

    for fold_dir in "${FOLD_DIRS[@]}"; do
        fold_name="$(basename "${fold_dir}")"
        fold_output_dir="${model_output_root}/${fold_name}"
        mkdir -p "${fold_output_dir}"
        args=(
            "--config" "${CONFIG_PATH}"
            "--train-manifest" "${fold_dir}/manifest_train.csv"
            "--val-manifest" "${fold_dir}/manifest_val.csv"
            "--test-manifest" "${fold_dir}/manifest_test.csv"
            "--root-dir" "${ROOT_DIR}"
            "--output-dir" "${fold_output_dir}"
            "--set" "finetune.fusion='eeg_only'"
            "--set" "finetune.contrastive_checkpoint_path=''"
            "--set" "finetune.eeg_baseline.enabled=true"
            "--set" "finetune.eeg_baseline.model_name='${model}'"
            "--set" "finetune.eeg_baseline.load_pretrained_weights=false"
            "--set" "finetune.eeg_baseline.checkpoint_path=''"
        )
        if [[ -n "${EPOCHS}" ]]; then
            args+=("--epochs" "${EPOCHS}")
        fi
        if [[ -n "${BATCH_SIZE}" ]]; then
            args+=("--batch-size" "${BATCH_SIZE}")
        fi
        if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
            args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
        fi
        if [[ -n "${NUM_WORKERS}" ]]; then
            args+=("--num-workers" "${NUM_WORKERS}")
        fi
        if [[ "${FORCE_CPU}" == "true" ]]; then
            args+=("--force-cpu")
        fi
        for extra_set in "${EXTRA_SET_ARGS[@]}"; do
            args+=("--set" "${extra_set}")
        done

        echo "  [${fold_name}] finetune"
        "${PYTHON}" "${REPO_ROOT}/run_finetune.py" "${args[@]}"
    done

    write_model_summary "${model_output_root}" >/dev/null
done

write_global_summary "${OUTPUT_ROOT}" >/dev/null
echo "Wrote summaries under ${OUTPUT_ROOT}"
