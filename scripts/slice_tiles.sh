#!/usr/bin/env bash

set -euo pipefail

# Wrapper around the poultry_geo CLI for the common "slice GeoTIFFs into square tiles"
# workflow. It accepts either a single GeoTIFF file or a directory of GeoTIFFs and
# forwards a file-level multiprocessing policy to the Python slicer.

usage() {
  cat <<'EOF'
Usage:
  slice_tiles.sh --input-dir PATH --output-dir PATH [options]

What it does:
  - slices one GeoTIFF or a directory tree of GeoTIFFs into square tiles
  - defaults to non-overlapping windows (`stride = tile-size`)
  - snaps the last window to the image edge instead of zero padding
  - can write TIFF, PNG, or JPG tiles
  - can process multiple GeoTIFFs in parallel, one full GeoTIFF per worker
  - appends every run to one project log file by default

Required:
  --input-dir PATH      GeoTIFF file or directory of GeoTIFFs to slice.
  --output-dir PATH     Directory where sliced tiles will be written.

Options:
  --tile-size N         Square tile size in pixels. Default: 256
  --stride N            Sliding-window stride in pixels. Default: tile size
  --format FMT          Output tile format: tif, png, jpg, jpeg. Default: png
  --jpeg-quality N      JPEG quality when format is jpg/jpeg. Default: 95
  --workers N|auto      Number of file-level worker processes. Default: auto
  --log-file PATH       Explicit log file path. Default: Code/command.log
  --no-log              Disable automatic append logging for this run
  --no-progress         Disable progress output for this run
  --python PATH         Python executable to use. Default: Code/.venv/bin/python, then python3
  -h, --help            Show this help text.

Output layout:
  - single input file: tiles are written directly into --output-dir
  - directory input: each GeoTIFF gets its own subdirectory under --output-dir

Examples:
  ./scripts/slice_tiles.sh \
    --input-dir ../Data/Test_ms_ga \
    --output-dir ./tmp/tiles_256_png \
    --tile-size 256 \
    --format png

  ./scripts/slice_tiles.sh \
    --input-dir ../Data/Test_ms_ga \
    --output-dir ./tmp/tiles_64_jpg \
    --tile-size 64 \
    --stride 32 \
    --format jpg \
    --jpeg-quality 92 \
    --workers 4

  ./scripts/slice_tiles.sh \
    --input-dir ../Data/Test_ms_ga \
    --output-dir ./tmp/tiles_256_tif \
    --tile-size 256 \
    --format tif \
    --workers auto
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"

DEFAULT_LOG_FILE="${CODE_DIR}/command.log"
INVOCATION_CWD="$(pwd)"
ORIGINAL_ARGS=("$@")
STARTED_SECONDS="${SECONDS}"

INPUT_DIR=""
OUTPUT_DIR=""
TILE_SIZE="256"
STRIDE=""
FORMAT="png"
JPEG_QUALITY="95"
WORKERS="auto"
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
    --input-dir=*)
      INPUT_DIR="${1#*=}"
      shift 1
      ;;
    --input-dir)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --input-dir requires a path." >&2
        exit 1
      fi
      INPUT_DIR="$2"
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
    --tile-size=*)
      TILE_SIZE="${1#*=}"
      shift 1
      ;;
    --tile-size)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --tile-size requires a value." >&2
        exit 1
      fi
      TILE_SIZE="$2"
      shift 2
      ;;
    --stride=*)
      STRIDE="${1#*=}"
      shift 1
      ;;
    --stride)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --stride requires a value." >&2
        exit 1
      fi
      STRIDE="$2"
      shift 2
      ;;
    --format=*)
      FORMAT="${1#*=}"
      shift 1
      ;;
    --format)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --format requires a value." >&2
        exit 1
      fi
      FORMAT="$2"
      shift 2
      ;;
    --jpeg-quality=*)
      JPEG_QUALITY="${1#*=}"
      shift 1
      ;;
    --jpeg-quality)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --jpeg-quality requires a value." >&2
        exit 1
      fi
      JPEG_QUALITY="$2"
      shift 2
      ;;
    --workers=*)
      WORKERS="${1#*=}"
      shift 1
      ;;
    --workers)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --workers requires a value." >&2
        exit 1
      fi
      WORKERS="$2"
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
    --no-progress)
      PROGRESS_ENABLED="0"
      shift 1
      ;;
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "--input-dir and --output-dir are required." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

COMMAND=(
  "${PYTHON_BIN}"
  -m
  poultry_geo
  slice
  "${INPUT_DIR}"
  "${OUTPUT_DIR}"
  --tile-size
  "${TILE_SIZE}"
  --format
  "${FORMAT}"
  --jpeg-quality
  "${JPEG_QUALITY}"
  --workers
  "${WORKERS}"
)

if [[ -n "${STRIDE}" ]]; then
  COMMAND+=(--stride "${STRIDE}")
fi

COMMAND+=(--no-log)
if [[ "${PROGRESS_ENABLED}" != "1" ]]; then
  COMMAND+=(--no-progress)
fi

cd "${CODE_DIR}"
echo "Running: ${COMMAND[*]}"
"${COMMAND[@]}"
