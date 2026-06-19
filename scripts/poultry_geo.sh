#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/poultry_geo.sh [wrapper-options] <command> [command-options]

What it does:
  - runs the Python poultry-geo CLI from Linux/macOS
  - appends every run to one project log file by default
  - handles virtual environment detection (.venv or conda)
  - forwards the remaining arguments directly to `scripts/poultry_geo.py`

Wrapper options:
  --python PATH         Python executable to use.
  --log-file PATH       Explicit log file path. Default: command.log
  --no-log              Disable automatic append logging for this run
  -h, --help            Show this wrapper help

Examples:
  ./scripts/poultry_geo.sh --help
  ./scripts/poultry_geo.sh slice ../Data/Test_ms_ga ./tmp/tiles --tile-size 256 --format png
  ./scripts/poultry_geo.sh ssl-grid-train "my-run" --config ssl_config.yaml
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

# First pass: extract wrapper-specific flags
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
    --no-log)
      LOG_ENABLED="0"
      shift 1
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#FORWARD_ARGS[@]} -eq 0 ]]; then
  usage
  exit 0
fi

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

setup_logging

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

PYTHON_SCRIPT="${SCRIPT_DIR}/poultry_geo.py"

"${PYTHON_BIN}" "${PYTHON_SCRIPT}" --no-log "${FORWARD_ARGS[@]}"
