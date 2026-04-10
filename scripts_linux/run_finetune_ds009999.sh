#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FINETUNE_CONFIG="configs/finetune_ds009999.yaml"
CACHE_ROOT="cache/ds009999"
OUTPUT_ROOT="outputs/ds009999"
EPOCHS="0"
BATCH_SIZE="0"
EVAL_BATCH_SIZE="0"
NUM_WORKERS="-1"
PYTHON_EXE=""
FORCE_CPU="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --finetune-config) FINETUNE_CONFIG="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
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

args=(
    "${REPO_ROOT}/run_finetune.py"
    "--config" "${FINETUNE_CONFIG}"
    "--loso"
    "--root-dir" "${CACHE_ROOT}"
    "--output-dir" "${OUTPUT_ROOT}/finetune"
)

if [[ "${EPOCHS}" != "0" ]]; then
    args+=("--epochs" "${EPOCHS}")
fi
if [[ "${BATCH_SIZE}" != "0" ]]; then
    args+=("--batch-size" "${BATCH_SIZE}")
fi
if [[ "${EVAL_BATCH_SIZE}" != "0" ]]; then
    args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
fi
if [[ "${NUM_WORKERS}" != "-1" ]]; then
    args+=("--num-workers" "${NUM_WORKERS}")
fi
if [[ "${FORCE_CPU}" == "true" ]]; then
    args+=("--force-cpu")
fi

"${PYTHON}" "${args[@]}"
