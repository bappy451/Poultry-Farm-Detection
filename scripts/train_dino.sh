#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_dino.sh --data-root PATH --output-dir PATH [options]

What it does:
  - builds DINO multi-crop train and validation loaders (2 global + 8 local crops)
  - trains a Domain-Adaptive DINO self-supervised Vision Transformer
  - updates EMA teacher with cosine momentum schedule and prototype centering vector
  - saves both full training checkpoint and extracted clean backbone weights
  - appends every run to one project log file by default

Options:
  --data-root PATH      Directory containing NAIP or RGB tiles. (Required)
  --output-dir PATH     Directory to store metrics.csv and checkpoint .pt files. (Required)
  --batch-size INT      Batch size for student/teacher crops. Default: 16.
  --model-name STR      ViT architecture size: S, B, L, or H. Default: B.
  --precision STR       fp32, fp16, bf16, or nf4. Default: fp32.
  --dino-out-dim INT    Prototype logit dimension K. Default: 65536.
  --epochs INT          Total training epochs. Default: 100.
  --resume              Resume training from last.pt if available in output-dir.
  --verbose             Enable rich multi-line progress reporting.

Examples:
  ./scripts/train_dino.sh --data-root ./tmp/test_jpg --output-dir ./runs/dino-smoke --batch-size 4 --model-name S --precision bf16 --device cuda --verbose
  ./scripts/train_dino.sh --data-root ./tmp/test_jpg --output-dir ./runs/dino-vitb --batch-size 16 --model-name B --epochs 100 --resume
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"
DEFAULT_LOG_FILE="${CODE_DIR}/command.log"
INVOCATION_CWD="$(pwd)"
ORIGINAL_ARGS=("$@")

PYTHON_BIN=""
LOG_FILE="${DEFAULT_LOG_FILE}"
LOG_ENABLED="1"
DATA_ROOT=""
OUTPUT_DIR=""
FORWARD_ARGS=()

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
  train-dino
  "${DATA_ROOT}"
  "${OUTPUT_DIR}"
  "${FORWARD_ARGS[@]}"
)

echo "Running: ${COMMAND[*]}"
"${COMMAND[@]}"
