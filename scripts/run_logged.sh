#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_logged.sh [options] -- command [args...]

What it does:
  - runs any shell command
  - captures stdout and stderr into one append-only log file
  - writes start time, end time, working directory, command text, and exit code
  - still streams live output to your terminal while the log is being written

Options:
  --log-file PATH       Explicit log file path. Default: Code/command.log
  -h, --help            Show this help text

Examples:
  ./scripts/run_logged.sh -- python3 -m poultry_geo --help

  ./scripts/run_logged.sh \
    -- python3 -m poultry_geo slice ../Data/Test_ms_ga ./tmp/tiles --tile-size 256 --format png

  ./scripts/run_logged.sh \
    --log-file ./tmp/manual.log \
    -- bash ./scripts/slice_tiles.sh --input-dir ../Data/Test_ms_ga --output-dir ./tmp/tiles_64 --tile-size 64
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LOG_FILE="${CODE_DIR}/command.log"

LOG_FILE="${DEFAULT_LOG_FILE}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "You must pass a command after --." >&2
  usage >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"
started_monotonic="${SECONDS}"

command_string=""
for token in "$@"; do
  if [[ -n "${command_string}" ]]; then
    command_string+=" "
  fi
  printf -v quoted_token '%q' "${token}"
  command_string+="${quoted_token}"
done

{
  echo
  echo "================================================================================"
  echo "Command Log"
  echo "================================================================================"
  echo "Started   : $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "Command   : ${command_string}"
  echo "CWD       : $(pwd)"
  echo "Log file  : ${LOG_FILE}"
  echo "================================================================================"
  echo
  echo "[logging] Appending command output to ${LOG_FILE}"
} | tee -a "${LOG_FILE}"

set +e
"$@" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

{
  echo
  elapsed_seconds=$((SECONDS - started_monotonic))
  printf -v duration '%02d:%02d:%02d' \
    "$((elapsed_seconds / 3600))" \
    "$(((elapsed_seconds % 3600) / 60))" \
    "$((elapsed_seconds % 60))"
  echo "[logging] Finished (exit=${status}, finished=$(date '+%Y-%m-%dT%H:%M:%S%z'), duration=${duration}, elapsed_seconds=${elapsed_seconds})"
} | tee -a "${LOG_FILE}"

exit "${status}"
