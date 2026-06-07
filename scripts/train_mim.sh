#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_mim.sh --data-root PATH --output-dir PATH [options]

What it does:
  - builds NAIP train and validation loaders
  - trains a MAE-style masked image modeling run
  - supports fp32, fp16, bf16, and nf4/QLoRA-style runs
  - appends every run to one project log file by default

Examples:
  ./scripts/train_mim.sh --python "$CONDA_PREFIX/bin/python" --data-root ./tmp/test_jpg --output-dir ./runs/mim-smoke --batch-size 4 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision fp16 --device mps
  ./scripts/train_mim.sh --python "$CONDA_PREFIX/bin/python" --data-root ./tmp/test_jpg --output-dir ./runs/mim-nf4 --batch-size 4 --crop-size 224 --channels 4 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision nf4 --use-lora --freeze-backbone
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LOG_FILE="${CODE_DIR}/command.log"
INVOCATION_CWD="$(pwd)"
ORIGINAL_ARGS=("$@")
STARTED_SECONDS="${SECONDS}"
LOG_FILE="${DEFAULT_LOG_FILE}"
LOG_ENABLED="1"
PYTHON_BIN=""
FORWARD_ARGS=()
DATA_ROOT=""
OUTPUT_DIR=""

setup_logging() {
  if [[ "${LOG_ENABLED}" != "1" ]]; then
    return
  fi

  mkdir -p "$(dirname "${LOG_FILE}")"

  local command_string=""
  local token=""
  for token in "$0" "${ORIGINAL_ARGS[@]}"; do
    if [[ -n "${command_string}" ]]; then
      command_string+=" "
    fi
    printf -v quoted_token '%q' "${token}"
    command_string+="${quoted_token}"
  done

  exec > >(tee -a "${LOG_FILE}") 2>&1

  echo
  echo "================================================================================"
  echo "Command Log"
  echo "================================================================================"
  echo "Started   : $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "Command   : ${command_string}"
  echo "CWD       : ${INVOCATION_CWD}"
  echo "Log file  : ${LOG_FILE}"
  echo "================================================================================"
  echo "[logging] Appending command output to ${LOG_FILE}"

  trap 'status=$?; elapsed_seconds=$((SECONDS - STARTED_SECONDS)); printf -v duration "%02d:%02d:%02d" "$((elapsed_seconds / 3600))" "$(((elapsed_seconds % 3600) / 60))" "$((elapsed_seconds % 60))"; echo "[logging] Finished (exit=${status}, finished=$(date "+%Y-%m-%dT%H:%M:%S%z"), duration=${duration}, elapsed_seconds=${elapsed_seconds})"' EXIT
}

scan_logging_flags() {
  local args=("$@")
  local index=0
  while [[ ${index} -lt ${#args[@]} ]]; do
    case "${args[${index}]}" in
      --no-log)
        LOG_ENABLED="0"
        ;;
      --log-file)
        if [[ $((index + 1)) -lt ${#args[@]} ]]; then
          LOG_FILE="${args[$((index + 1))]}"
          index=$((index + 1))
        fi
        ;;
      --log-file=*)
        LOG_FILE="${args[${index}]#--log-file=}"
        ;;
    esac
    index=$((index + 1))
  done
}

scan_logging_flags "${ORIGINAL_ARGS[@]}"
setup_logging

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#--python=}"
      shift 1
      ;;
    --log-file)
      shift 2
      ;;
    --log-file=*)
      shift 1
      ;;
    --no-log)
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ -z "${DATA_ROOT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "--data-root and --output-dir are required." >&2
  usage >&2
  exit 2
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${CODE_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${CODE_DIR}/.venv/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/python.exe" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/python.exe"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="python3"
  fi
fi

COMMAND=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/scripts/poultry_geo.py"
  --no-log
  train-mim
  "${DATA_ROOT}"
  "${OUTPUT_DIR}"
  "${FORWARD_ARGS[@]}"
)

echo "Running: ${COMMAND[*]}"
"${COMMAND[@]}"
