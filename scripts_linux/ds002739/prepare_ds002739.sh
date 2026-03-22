#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DS_ROOT="../ds002739"
OUTPUT_ROOT="cache/ds002739"
SUBJECTS=()
RUNS=()
NUM_WORKERS="2"
SPLIT_MODE="loso"
TRAIN_SUBJECTS="21"
VAL_SUBJECTS="2"
TEST_SUBJECTS="1"
EEG_WINDOW_SEC="4.0"
FMRI_WINDOW_SEC="4.0"
EEG_SEQ_LEN="0"
TRAINING_READY="true"
EEG_ONLY="true"
TARGET_CHANNEL_MANIFEST=""
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
        --ds-root) DS_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --subjects) mapfile -t SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --runs) mapfile -t RUNS < <(parse_array_arg "$2"); shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --split-mode) SPLIT_MODE="$2"; shift 2 ;;
        --train-subjects) TRAIN_SUBJECTS="$2"; shift 2 ;;
        --val-subjects) VAL_SUBJECTS="$2"; shift 2 ;;
        --test-subjects) TEST_SUBJECTS="$2"; shift 2 ;;
        --eeg-window-sec) EEG_WINDOW_SEC="$2"; shift 2 ;;
        --fmri-window-sec) FMRI_WINDOW_SEC="$2"; shift 2 ;;
        --eeg-seq-len) EEG_SEQ_LEN="$2"; shift 2 ;;
        --training-ready) TRAINING_READY="true"; shift ;;
        --no-training-ready) TRAINING_READY="false"; shift ;;
        --eeg-only) EEG_ONLY="true"; shift ;;
        --no-eeg-only) EEG_ONLY="false"; shift ;;
        --target-channel-manifest) TARGET_CHANNEL_MANIFEST="$2"; shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
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

resolved_eeg_seq_len="${EEG_SEQ_LEN}"
if [[ "${resolved_eeg_seq_len}" == "0" ]]; then
    resolved_eeg_seq_len="$(python - <<PY
import math
print(max(1,int(round(float("${EEG_WINDOW_SEC}")))))
PY
)"
fi

if [[ "${SPLIT_MODE}" != "none" && ${#SUBJECTS[@]} -gt 0 ]]; then
    if [[ "${SPLIT_MODE}" == "loso" ]]; then
        required_subjects=$((VAL_SUBJECTS + 1))
    else
        required_subjects=$((TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS))
    fi
    if [[ ${#SUBJECTS[@]} -lt ${required_subjects} ]]; then
        echo "Provided subject subset is smaller than the requested split sizes; disabling split generation for this run." >&2
        SPLIT_MODE="none"
    fi
fi

cd "${REPO_ROOT}"

CLI_ARGS=(
    "preprocess/prepare_ds002739.py"
    "--ds-root" "${DS_ROOT}"
    "--output-root" "${OUTPUT_ROOT}"
    "--fmri-mode" "volume"
    "--fmri-voxel-size" "2.0" "2.0" "2.0"
    "--pack-subject-files"
    "--eeg-window-sec" "${EEG_WINDOW_SEC}"
    "--fmri-window-sec" "${FMRI_WINDOW_SEC}"
    "--eeg-seq-len" "${resolved_eeg_seq_len}"
    "--eeg-patch-len" "200"
    "--eeg-target-sfreq" "200"
    "--eeg-lfreq" "0.5"
    "--eeg-hfreq" "40"
    "--tr" "2.0"
    "--fmri-max-shape" "48" "48" "48"
    "--num-workers" "${NUM_WORKERS}"
    "--split-mode" "${SPLIT_MODE}"
    "--train-subjects" "${TRAIN_SUBJECTS}"
    "--val-subjects" "${VAL_SUBJECTS}"
    "--test-subjects" "${TEST_SUBJECTS}"
)

if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--subjects" "${SUBJECTS[@]}")
fi
if [[ ${#RUNS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--runs" "${RUNS[@]}")
fi

if [[ "${TRAINING_READY}" == "true" ]]; then
    CLI_ARGS+=("--training-ready")
else
    CLI_ARGS+=("--no-training-ready")
fi

if [[ "${EEG_ONLY}" == "true" ]]; then
    CLI_ARGS+=("--eeg-only")
fi

if [[ -n "${TARGET_CHANNEL_MANIFEST}" ]]; then
    CLI_ARGS+=("--target-channel-manifest" "${TARGET_CHANNEL_MANIFEST}")
fi

if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    echo "Preparing ds002739 for subjects: ${SUBJECTS[*]}"
else
    echo "Preparing ds002739 for all subjects..."
fi
if [[ ${#RUNS[@]} -gt 0 ]]; then
    echo "Restricting to runs: ${RUNS[*]}"
fi

echo "Output root: ${OUTPUT_ROOT}"
echo "Worker processes: ${NUM_WORKERS}"
echo "Split mode: ${SPLIT_MODE}"
echo "Training ready: ${TRAINING_READY}"
echo "fMRI preprocessing: resample to 2.0x2.0x2.0 mm, then center-crop to 48x48x48"

"${PYTHON}" "${CLI_ARGS[@]}"
