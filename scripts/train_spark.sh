#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_spark.sh --data-root PATH --output-dir PATH [options]

What it does:
  - builds NAIP train and validation loaders
  - trains a Sparse Masked Modeling (SparK) SSL run
  - supports hierarchical CNN backbones (ResNet50, YOLO CSPDarkNet)
  - appends every run to one project log file by default

Options:
  --verbose             Enable rich multi-line progress reporting with live metrics.

Examples:
  ./scripts/train_spark.sh --python "$CONDA_PREFIX/bin/python" --data-root ./tmp/test_jpg --output-dir ./runs/spark-smoke --batch-size 4 --backbone-type resnet50 --precision bf16 --device cuda --verbose
  ./scripts/train_spark.sh --python "$CONDA_PREFIX/bin/python" --data-root ./tmp/test_jpg --output-dir ./runs/spark-yolo --batch-size 4 --backbone-type cspdarknet --mask-ratio 0.75 --epochs 10 --verbose
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"
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

  if [[ "${POULTRY_GEO_LOG_WRAPPED:-0}" != "1" ]]; then
    export POULTRY_GEO_LOG_WRAPPED="1"
    set +e
    "${BASH:-bash}" "$0" "${ORIGINAL_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    local wrapped_status=${PIPESTATUS[0]}
    set -e
    exit "${wrapped_status}"
  fi

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
    --python=*)
      PYTHON_BIN="${1#--python=}"
      shift 1
      ;;
    --python)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --python requires a path." >&2
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --log-file=*)
      LOG_FILE="${1#--log-file=}"
      shift 1
      ;;
    --log-file)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --log-file requires a path." >&2
        exit 1
      fi
      LOG_FILE="$2"
      shift 2
      ;;
    --no-log)
      LOG_ENABLED="0"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --data-root=*)
      DATA_ROOT="${1#*=}"
      shift 1
      ;;
    --data-root)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --data-root requires a path." >&2
        exit 1
      fi
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift 1
      ;;
    --output-dir)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --output-dir requires a path." >&2
        exit 1
      fi
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

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

COMMAND=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/scripts/poultry_geo.py"
  --no-log
  train-spark
  "${DATA_ROOT}"
  "${OUTPUT_DIR}"
  "${FORWARD_ARGS[@]}"
)

echo "Running: ${COMMAND[*]}"
"${COMMAND[@]}"
