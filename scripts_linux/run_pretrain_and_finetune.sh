#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAIN_DATASETS=("ds002336" "ds002338" "ds002739")
TARGET_DATASET="ds002739"
JOINT_TRAIN_CONFIG="configs/train_joint_contrastive.yaml"
DS002336_FINETUNE_CONFIG="configs/finetune_ds002336.yaml"
DS002338_FINETUNE_CONFIG="configs/finetune_ds002338.yaml"
DS002739_FINETUNE_CONFIG="configs/finetune_ds002739.yaml"
DS002336_ROOT="../ds002336"
DS002338_ROOT="../ds002338"
DS002739_ROOT="../ds002739"
JOINT_CACHE_ROOT="cache/joint_contrastive"
DS002336_CACHE_ROOT="cache/ds002336"
DS002338_CACHE_ROOT="cache/ds002338"
DS002739_CACHE_ROOT="cache/ds002739"
JOINT_OUTPUT_ROOT="outputs/joint_contrastive"
DS002336_OUTPUT_ROOT="outputs/ds002336"
DS002338_OUTPUT_ROOT="outputs/ds002338"
DS002739_OUTPUT_ROOT="outputs/ds002739"
PRETRAINED_WEIGHTS_DIR="pretrained_weights"
JOINT_EEG_WINDOW_SEC="8.0"
PRETRAIN_EPOCHS="0"
FINETUNE_EPOCHS="0"
PRETRAIN_BATCH_SIZE="0"
FINETUNE_BATCH_SIZE="0"
BATCH_SIZE="0"
EVAL_BATCH_SIZE="0"
NUM_WORKERS="-1"
GPU_COUNT="1"
GPU_IDS=""
SKIP_PRETRAIN="false"
SKIP_FINETUNE="false"
TEST_ONLY="false"
FORCE_CPU="false"
PYTHON_EXE=""

parse_array_arg() {
    local raw="$1"
    IFS=',' read -r -a _arr <<< "$raw"
    for _item in "${_arr[@]}"; do
        if [[ -n "${_item}" ]]; then
            echo "${_item}"
        fi
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrain-datasets) mapfile -t PRETRAIN_DATASETS < <(parse_array_arg "$2"); shift 2 ;;
        --target-dataset) TARGET_DATASET="$2"; shift 2 ;;
        --joint-train-config) JOINT_TRAIN_CONFIG="$2"; shift 2 ;;
        --ds002336-finetune-config) DS002336_FINETUNE_CONFIG="$2"; shift 2 ;;
        --ds002338-finetune-config) DS002338_FINETUNE_CONFIG="$2"; shift 2 ;;
        --ds002739-finetune-config) DS002739_FINETUNE_CONFIG="$2"; shift 2 ;;
        --ds002336-root) DS002336_ROOT="$2"; shift 2 ;;
        --ds002338-root) DS002338_ROOT="$2"; shift 2 ;;
        --ds002739-root) DS002739_ROOT="$2"; shift 2 ;;
        --joint-cache-root) JOINT_CACHE_ROOT="$2"; shift 2 ;;
        --ds002336-cache-root) DS002336_CACHE_ROOT="$2"; shift 2 ;;
        --ds002338-cache-root) DS002338_CACHE_ROOT="$2"; shift 2 ;;
        --ds002739-cache-root) DS002739_CACHE_ROOT="$2"; shift 2 ;;
        --joint-output-root) JOINT_OUTPUT_ROOT="$2"; shift 2 ;;
        --ds002336-output-root) DS002336_OUTPUT_ROOT="$2"; shift 2 ;;
        --ds002338-output-root) DS002338_OUTPUT_ROOT="$2"; shift 2 ;;
        --ds002739-output-root) DS002739_OUTPUT_ROOT="$2"; shift 2 ;;
        --pretrained-weights-dir) PRETRAINED_WEIGHTS_DIR="$2"; shift 2 ;;
        --joint-eeg-window-sec) JOINT_EEG_WINDOW_SEC="$2"; shift 2 ;;
        --pretrain-epochs) PRETRAIN_EPOCHS="$2"; shift 2 ;;
        --finetune-epochs) FINETUNE_EPOCHS="$2"; shift 2 ;;
        --pretrain-batch-size) PRETRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --finetune-batch-size) FINETUNE_BATCH_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --skip-pretrain) SKIP_PRETRAIN="true"; shift ;;
        --skip-finetune) SKIP_FINETUNE="true"; shift ;;
        --test-only) TEST_ONLY="true"; shift ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

contains_item() {
    local needle="$1"
    shift
    for item in "$@"; do
        if [[ "${item}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

invoke_or_throw() {
    local step_name="$1"
    shift
    "$@"
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        echo "${step_name} failed with exit code ${code}" >&2
        exit ${code}
    fi
}

config_uses_eeg_baseline() {
    local config_path="$1"
    "${PYTHON}" - <<'PY' "${config_path}"
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
enabled = bool((((payload.get("finetune") or {}).get("eeg_baseline") or {}).get("enabled", False)))
print("true" if enabled else "false")
PY
}

test_joint_cache_ready() {
    local cache_root="$1"
    [[ -f "${cache_root}/manifest_all.csv" ]]
}

test_target_cache_ready() {
    local cache_root="$1"
    local loso_root="${cache_root}/loso_subjectwise"
    if [[ ! -d "${loso_root}" ]]; then
        return 1
    fi
    shopt -s nullglob
    local folds=("${loso_root}"/fold_*)
    shopt -u nullglob
    if [[ ${#folds[@]} -eq 0 ]]; then
        return 1
    fi
    local fold
    for fold in "${folds[@]}"; do
        [[ -f "${fold}/manifest_train.csv" && -f "${fold}/manifest_val.csv" && -f "${fold}/manifest_test.csv" ]] || return 1
    done
    return 0
}

if [[ "${SKIP_PRETRAIN}" == "true" && "${SKIP_FINETUNE}" == "true" ]]; then
    echo "--skip-pretrain and --skip-finetune cannot both be set" >&2
    exit 2
fi
if [[ ${GPU_COUNT} -le 0 ]]; then
    echo "--gpu-count must be >= 1" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
NONEMPTY_GPU_IDS=()
for item in "${GPU_ID_LIST[@]}"; do
    trimmed="$(echo "${item}" | xargs)"
    if [[ -n "${trimmed}" ]]; then
        NONEMPTY_GPU_IDS+=("${trimmed}")
    fi
done
if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 && ${#NONEMPTY_GPU_IDS[@]} -lt ${GPU_COUNT} ]]; then
    echo "--gpu-ids count (${#NONEMPTY_GPU_IDS[@]}) must be >= --gpu-count (${GPU_COUNT})" >&2
    exit 2
fi
if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${NONEMPTY_GPU_IDS[*]}")"
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

USE_MULTI_GPU="false"
if [[ "${FORCE_CPU}" != "true" && ${GPU_COUNT} -gt 1 ]]; then
    USE_MULTI_GPU="true"
fi

if [[ ${PRETRAIN_BATCH_SIZE} -le 0 && ${BATCH_SIZE} -gt 0 ]]; then PRETRAIN_BATCH_SIZE="${BATCH_SIZE}"; fi
if [[ ${FINETUNE_BATCH_SIZE} -le 0 && ${BATCH_SIZE} -gt 0 ]]; then FINETUNE_BATCH_SIZE="${BATCH_SIZE}"; fi
if [[ "${TEST_ONLY}" == "true" && "${SKIP_FINETUNE}" == "true" ]]; then
    echo "--test-only requires finetune stage" >&2
    exit 2
fi

if [[ -n "${PYTHON_EXE}" ]]; then
    PYTHON="${PYTHON_EXE}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="python"
fi

cd "${REPO_ROOT}"

resolve_dataset_root() {
    local current_root="$1"
    local dataset_name="$2"
    if [[ -d "${current_root}" ]]; then
        echo "${current_root}"
        return
    fi
    local candidates=(
        "data/${dataset_name}"
        "../data/${dataset_name}"
        "../${dataset_name}"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done
    echo "${current_root}"
}

DS002336_ROOT="$(resolve_dataset_root "${DS002336_ROOT}" "ds002336")"
DS002338_ROOT="$(resolve_dataset_root "${DS002338_ROOT}" "ds002338")"
DS002739_ROOT="$(resolve_dataset_root "${DS002739_ROOT}" "ds002739")"

joint_manifest_path="${JOINT_CACHE_ROOT}/manifest_all.csv"
joint_channel_manifest="${JOINT_CACHE_ROOT}/eeg_channels_target.csv"
joint_training_checkpoint_path="${JOINT_OUTPUT_ROOT}/contrastive/checkpoints/best.pth"
joint_checkpoint_path="${PRETRAINED_WEIGHTS_DIR}/contrastive_best.pth"
joint_checkpoint_source_path=""

if [[ "${SKIP_PRETRAIN}" != "true" ]]; then
    if test_joint_cache_ready "${JOINT_CACHE_ROOT}"; then
        echo "Found existing joint cache under ${JOINT_CACHE_ROOT}; skipping preprocessing."
    else
        joint_prepare_args=(
            "${REPO_ROOT}/scripts_linux/prepare_joint_contrastive.sh"
            "--output-root" "${JOINT_CACHE_ROOT}"
            "--eeg-window-sec" "${JOINT_EEG_WINDOW_SEC}"
            "--python-exe" "${PYTHON}"
        )
        if [[ ${#PRETRAIN_DATASETS[@]} -gt 0 ]]; then
            joint_prepare_args+=("--datasets" "$(IFS=,; echo "${PRETRAIN_DATASETS[*]}")")
        fi
        if [[ ${NUM_WORKERS} -ge 1 ]]; then
            joint_prepare_args+=("--num-workers" "${NUM_WORKERS}")
        fi
        joint_prepare_args+=(
            "--ds002336-root" "${DS002336_ROOT}"
            "--ds002338-root" "${DS002338_ROOT}"
            "--ds002739-root" "${DS002739_ROOT}"
        )

        echo "Preparing joint contrastive cache..."
        invoke_or_throw "joint preprocessing" "${joint_prepare_args[@]}"
    fi

    train_args=(
        "--config" "${JOINT_TRAIN_CONFIG}"
        "--manifest" "${joint_manifest_path}"
        "--root-dir" "${JOINT_CACHE_ROOT}"
        "--output-dir" "${JOINT_OUTPUT_ROOT}/contrastive"
    )
    if [[ ${PRETRAIN_EPOCHS} -gt 0 ]]; then train_args+=("--epochs" "${PRETRAIN_EPOCHS}"); fi
    if [[ ${PRETRAIN_BATCH_SIZE} -gt 0 ]]; then train_args+=("--batch-size" "${PRETRAIN_BATCH_SIZE}"); fi
    if [[ ${NUM_WORKERS} -ge 0 ]]; then train_args+=("--num-workers" "${NUM_WORKERS}"); fi
    if [[ "${FORCE_CPU}" == "true" ]]; then train_args+=("--force-cpu"); fi
    if [[ "${FORCE_CPU}" != "true" ]]; then
        train_args+=("--set" "train.gpu_count=${GPU_COUNT}")
        if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 ]]; then
            train_args+=("--set" "train.gpu_ids=$(IFS=,; echo "${NONEMPTY_GPU_IDS[*]}")")
        fi
    fi

    echo "Running joint contrastive pretraining..."
    if [[ "${USE_MULTI_GPU}" == "true" ]]; then
        invoke_or_throw "joint pretraining" "${PYTHON}" -m torch.distributed.run --nproc_per_node "${GPU_COUNT}" "${REPO_ROOT}/run_train.py" "${train_args[@]}"
    else
        invoke_or_throw "joint pretraining" "${PYTHON}" "${REPO_ROOT}/run_train.py" "${train_args[@]}"
    fi

    if [[ ! -f "${joint_training_checkpoint_path}" ]]; then
        echo "Pretrain checkpoint not found after pretraining: ${joint_training_checkpoint_path}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${joint_checkpoint_path}")"
    cp -f "${joint_training_checkpoint_path}" "${joint_checkpoint_path}"
    echo "Synced pretrain best checkpoint to: ${joint_checkpoint_path}"
fi

if [[ -f "${joint_checkpoint_path}" ]]; then
    joint_checkpoint_source_path="${joint_checkpoint_path}"
elif [[ -f "${joint_training_checkpoint_path}" ]]; then
    joint_checkpoint_source_path="${joint_training_checkpoint_path}"
    echo "Using fallback pretrain checkpoint path: ${joint_checkpoint_source_path}"
fi

if [[ "${SKIP_FINETUNE}" == "true" ]]; then
    exit 0
fi

if [[ "${TARGET_DATASET}" == "ds002336" || "${TARGET_DATASET}" == "ds002338" ]]; then
    target_prepare_script="${REPO_ROOT}/scripts_linux/ds00233x/prepare_ds00233x.sh"
else
    target_prepare_script="${REPO_ROOT}/scripts_linux/ds002739/prepare_ds002739.sh"
fi

if [[ "${TARGET_DATASET}" == "ds002336" ]]; then
    target_finetune_config="${DS002336_FINETUNE_CONFIG}"
    target_cache_root="${DS002336_CACHE_ROOT}"
    target_output_root="${DS002336_OUTPUT_ROOT}"
elif [[ "${TARGET_DATASET}" == "ds002338" ]]; then
    target_finetune_config="${DS002338_FINETUNE_CONFIG}"
    target_cache_root="${DS002338_CACHE_ROOT}"
    target_output_root="${DS002338_OUTPUT_ROOT}"
else
    target_finetune_config="${DS002739_FINETUNE_CONFIG}"
    target_cache_root="${DS002739_CACHE_ROOT}"
    target_output_root="${DS002739_OUTPUT_ROOT}"
fi

if test_target_cache_ready "${target_cache_root}"; then
    echo "Found existing target cache under ${target_cache_root}; skipping preprocessing."
else
    target_prepare_args=(
        "${target_prepare_script}"
        "--output-root" "${target_cache_root}"
        "--split-mode" "loso"
        "--training-ready"
        "--eeg-only"
        "--python-exe" "${PYTHON}"
    )
    if [[ -f "${joint_channel_manifest}" ]]; then
        target_prepare_args+=("--target-channel-manifest" "${joint_channel_manifest}")
    fi
    if [[ "${TARGET_DATASET}" == "ds002336" ]]; then
        target_prepare_args+=("--dataset-name" "ds002336" "--ds-root" "${DS002336_ROOT}")
    elif [[ "${TARGET_DATASET}" == "ds002338" ]]; then
        target_prepare_args+=("--dataset-name" "ds002338" "--ds-root" "${DS002338_ROOT}")
    else
        target_prepare_args+=("--ds-root" "${DS002739_ROOT}")
    fi

    echo "Preparing target finetune cache for ${TARGET_DATASET}..."
    invoke_or_throw "target preprocessing" "${target_prepare_args[@]}"
fi

loso_dir="${target_cache_root}/loso_subjectwise"
shopt -s nullglob
fold_dirs=("${loso_dir}"/fold_*)
shopt -u nullglob
if [[ ${#fold_dirs[@]} -eq 0 ]]; then
    echo "No LOSO fold directories found under ${loso_dir}" >&2
    exit 1
fi

finetune_root="${target_output_root}/finetune"
mkdir -p "${finetune_root}"

baseline_enabled="$(config_uses_eeg_baseline "${target_finetune_config}")"

if [[ -n "${joint_checkpoint_source_path}" && "${baseline_enabled}" != "true" ]]; then
    echo "Using pretrain best checkpoint for finetune from: ${joint_checkpoint_source_path}"
elif [[ "${baseline_enabled}" != "true" ]]; then
    echo "Pretrain checkpoint not found at expected paths: ${joint_checkpoint_path} or ${joint_training_checkpoint_path}; finetune will run without contrastive checkpoint unless you pass --contrastive-checkpoint manually."
fi

finetune_args=(
    "--config" "${target_finetune_config}"
    "--loso"
    "--root-dir" "${target_cache_root}"
    "--output-dir" "${finetune_root}"
)
if [[ -n "${joint_checkpoint_source_path}" && "${baseline_enabled}" != "true" ]]; then
    finetune_args+=("--contrastive-checkpoint" "${joint_checkpoint_source_path}")
fi
if [[ "${TEST_ONLY}" == "true" ]]; then
    finetune_args+=("--test-only")
fi
if [[ ${FINETUNE_EPOCHS} -gt 0 ]]; then finetune_args+=("--epochs" "${FINETUNE_EPOCHS}"); fi
if [[ ${FINETUNE_BATCH_SIZE} -gt 0 ]]; then finetune_args+=("--batch-size" "${FINETUNE_BATCH_SIZE}"); fi
if [[ ${EVAL_BATCH_SIZE} -gt 0 ]]; then finetune_args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}"); fi
if [[ ${NUM_WORKERS} -ge 0 ]]; then finetune_args+=("--num-workers" "${NUM_WORKERS}"); fi
if [[ "${FORCE_CPU}" == "true" ]]; then finetune_args+=("--force-cpu"); fi
if [[ "${FORCE_CPU}" != "true" ]]; then
    finetune_args+=("--set" "train.gpu_count=${GPU_COUNT}")
    if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 ]]; then
        finetune_args+=("--set" "train.gpu_ids=$(IFS=,; echo "${NONEMPTY_GPU_IDS[*]}")")
    fi
fi

echo "Running LOSO finetune..."
if [[ "${USE_MULTI_GPU}" == "true" ]]; then
    invoke_or_throw "LOSO finetune" "${PYTHON}" -m torch.distributed.run --nproc_per_node "${GPU_COUNT}" "${REPO_ROOT}/run_finetune.py" "${finetune_args[@]}"
else
    invoke_or_throw "LOSO finetune" "${PYTHON}" "${REPO_ROOT}/run_finetune.py" "${finetune_args[@]}"
fi
