#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  inspect_segmentation.sh --data-root PATH [options]

What it does:
  - loads one NAIP batch from nested image tiles
  - adapts n-band imagery to 3 channels
  - runs the ViT backbone with optional LoRA adapters
  - decodes patch embeddings back into a binary roof mask
  - appends every run to one project log file by default

Required:
  --data-root PATH      Root directory containing nested JPG, PNG, TIFF, or NPY tiles.

Options:
  --batch-size N                Batch size. Default: 2
  --crop-size N                 Crop size and mask size. Default: 256
  --channels N                  Expected input channel count. Default: infer
  --precision MODE              fp32, fp16, bf16, nf4. Default: fp32
  --patch-size N                ViT patch size. Default: 16
  --embed-dim N                 ViT embedding dimension. Default: 192
  --depth N                     Number of transformer blocks. Default: 8
  --num-heads N                 Attention heads. Default: 6
  --mlp-ratio N                 Feed-forward expansion ratio. Default: 4.0
  --num-workers N               DataLoader workers. Default: 4
  --device NAME                 cpu, gpu, cuda:0, or mps. Default: auto
  --gradient-checkpointing      Enable checkpointed transformer blocks
  --use-lora                    Inject LoRA adapters into ViT attention projections
  --freeze-backbone             Freeze non-LoRA backbone parameters
  --lora-rank N                 LoRA rank. Default: 4
  --lora-alpha N                LoRA scaling alpha. Default: 8.0
  --lora-dropout N              LoRA dropout. Default: 0.0
  --decoder-hidden-channels N   Decoder hidden channels. Default: 128
  --python PATH                 Python executable to use. Default: Code/.venv/bin/python, then conda env, then python3
  --log-file PATH               Explicit log file path. Default: Code/command.log
  --no-log                      Disable automatic append logging for this run
  --no-progress                 Disable progress output for this run
  -h, --help                    Show this help text.

Examples:
  ./scripts/inspect_segmentation.sh \
    --data-root ./tmp/test_jpg \
    --batch-size 2 \
    --crop-size 256 \
    --channels 3 \
    --patch-size 16 \
    --embed-dim 192 \
    --depth 8 \
    --num-heads 6 \
    --use-lora \
    --freeze-backbone

  ./scripts/inspect_segmentation.sh \
    --data-root ./tmp/test_jpg \
    --batch-size 2 \
    --crop-size 512 \
    --channels 4 \
    --precision bf16 \
    --patch-size 16 \
    --embed-dim 384 \
    --depth 12 \
    --num-heads 6 \
    --use-lora \
    --lora-rank 8 \
    --decoder-hidden-channels 128 \
    --device gpu
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"

DEFAULT_LOG_FILE="${CODE_DIR}/command.log"
INVOCATION_CWD="$(pwd)"
ORIGINAL_ARGS=("$@")
STARTED_SECONDS="${SECONDS}"

DATA_ROOT=""
BATCH_SIZE="2"
CROP_SIZE="256"
CHANNELS=""
PRECISION="fp32"
PATCH_SIZE="16"
EMBED_DIM="192"
DEPTH="8"
NUM_HEADS="6"
MLP_RATIO="4.0"
NUM_WORKERS="4"
DEVICE_NAME=""
ENABLE_CHECKPOINTING="0"
USE_LORA="0"
FREEZE_BACKBONE="0"
LORA_RANK="4"
LORA_ALPHA="8.0"
LORA_DROPOUT="0.0"
DECODER_HIDDEN_CHANNELS="128"
LOG_FILE="${DEFAULT_LOG_FILE}"
LOG_ENABLED="1"
PYTHON_BIN=""
PROGRESS_ENABLED="1"

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
    --data-root=*) DATA_ROOT="${1#*=}"; shift 1 ;;
    --data-root)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --data-root requires a path." >&2
        exit 1
      fi
      DATA_ROOT="$2"; shift 2 ;;
    --batch-size=*) BATCH_SIZE="${1#*=}"; shift 1 ;;
    --batch-size)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --batch-size requires a value." >&2
        exit 1
      fi
      BATCH_SIZE="$2"; shift 2 ;;
    --crop-size=*) CROP_SIZE="${1#*=}"; shift 1 ;;
    --crop-size)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --crop-size requires a value." >&2
        exit 1
      fi
      CROP_SIZE="$2"; shift 2 ;;
    --channels=*) CHANNELS="${1#*=}"; shift 1 ;;
    --channels)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --channels requires a value." >&2
        exit 1
      fi
      CHANNELS="$2"; shift 2 ;;
    --precision=*) PRECISION="${1#*=}"; shift 1 ;;
    --precision)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --precision requires a value." >&2
        exit 1
      fi
      PRECISION="$2"; shift 2 ;;
    --patch-size=*) PATCH_SIZE="${1#*=}"; shift 1 ;;
    --patch-size)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --patch-size requires a value." >&2
        exit 1
      fi
      PATCH_SIZE="$2"; shift 2 ;;
    --embed-dim=*) EMBED_DIM="${1#*=}"; shift 1 ;;
    --embed-dim)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --embed-dim requires a value." >&2
        exit 1
      fi
      EMBED_DIM="$2"; shift 2 ;;
    --depth=*) DEPTH="${1#*=}"; shift 1 ;;
    --depth)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --depth requires a value." >&2
        exit 1
      fi
      DEPTH="$2"; shift 2 ;;
    --num-heads=*) NUM_HEADS="${1#*=}"; shift 1 ;;
    --num-heads)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --num-heads requires a value." >&2
        exit 1
      fi
      NUM_HEADS="$2"; shift 2 ;;
    --mlp-ratio=*) MLP_RATIO="${1#*=}"; shift 1 ;;
    --mlp-ratio)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --mlp-ratio requires a value." >&2
        exit 1
      fi
      MLP_RATIO="$2"; shift 2 ;;
    --num-workers=*) NUM_WORKERS="${1#*=}"; shift 1 ;;
    --num-workers)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --num-workers requires a value." >&2
        exit 1
      fi
      NUM_WORKERS="$2"; shift 2 ;;
    --device=*) DEVICE_NAME="${1#*=}"; shift 1 ;;
    --device)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --device requires a value." >&2
        exit 1
      fi
      DEVICE_NAME="$2"; shift 2 ;;
    --gradient-checkpointing) ENABLE_CHECKPOINTING="1"; shift 1 ;;
    --use-lora) USE_LORA="1"; shift 1 ;;
    --freeze-backbone) FREEZE_BACKBONE="1"; shift 1 ;;
    --lora-rank=*) LORA_RANK="${1#*=}"; shift 1 ;;
    --lora-rank)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --lora-rank requires a value." >&2
        exit 1
      fi
      LORA_RANK="$2"; shift 2 ;;
    --lora-alpha=*) LORA_ALPHA="${1#*=}"; shift 1 ;;
    --lora-alpha)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --lora-alpha requires a value." >&2
        exit 1
      fi
      LORA_ALPHA="$2"; shift 2 ;;
    --lora-dropout=*) LORA_DROPOUT="${1#*=}"; shift 1 ;;
    --lora-dropout)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --lora-dropout requires a value." >&2
        exit 1
      fi
      LORA_DROPOUT="$2"; shift 2 ;;
    --decoder-hidden-channels=*) DECODER_HIDDEN_CHANNELS="${1#*=}"; shift 1 ;;
    --decoder-hidden-channels)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --decoder-hidden-channels requires a value." >&2
        exit 1
      fi
      DECODER_HIDDEN_CHANNELS="$2"; shift 2 ;;
    --log-file=*) LOG_FILE="${1#--log-file=}"; shift 1 ;;
    --log-file)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --log-file requires a path." >&2
        exit 1
      fi
      LOG_FILE="$2"; shift 2 ;;
    --no-log) LOG_ENABLED="0"; shift 1 ;;
    --no-progress) PROGRESS_ENABLED="0"; shift 1 ;;
    --python=*) PYTHON_BIN="${1#--python=}"; shift 1 ;;
    --python)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --python requires a path." >&2
        exit 1
      fi
      PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "--data-root is required." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

COMMAND=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/scripts/poultry_geo.py"
  --no-log
  inspect-segmentation
  "${DATA_ROOT}"
  --batch-size
  "${BATCH_SIZE}"
  --crop-size
  "${CROP_SIZE}"
  --precision
  "${PRECISION}"
  --patch-size
  "${PATCH_SIZE}"
  --embed-dim
  "${EMBED_DIM}"
  --depth
  "${DEPTH}"
  --num-heads
  "${NUM_HEADS}"
  --mlp-ratio
  "${MLP_RATIO}"
  --num-workers
  "${NUM_WORKERS}"
  --lora-rank
  "${LORA_RANK}"
  --lora-alpha
  "${LORA_ALPHA}"
  --lora-dropout
  "${LORA_DROPOUT}"
  --decoder-hidden-channels
  "${DECODER_HIDDEN_CHANNELS}"
)

if [[ -n "${CHANNELS}" ]]; then
  COMMAND+=(--channels "${CHANNELS}")
fi

if [[ -n "${DEVICE_NAME}" ]]; then
  COMMAND+=(--device "${DEVICE_NAME}")
fi

if [[ "${ENABLE_CHECKPOINTING}" == "1" ]]; then
  COMMAND+=(--gradient-checkpointing)
fi

if [[ "${USE_LORA}" == "1" ]]; then
  COMMAND+=(--use-lora)
fi

if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  COMMAND+=(--freeze-backbone)
fi

if [[ "${PROGRESS_ENABLED}" != "1" ]]; then
  COMMAND+=(--no-progress)
fi

cd "${CODE_DIR}"
echo "Running: ${COMMAND[*]}"
"${COMMAND[@]}"
