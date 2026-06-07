from __future__ import annotations

import argparse
import io
import os
import shlex
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence


PROJECT_CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_COMMAND_LOG_FILE = PROJECT_CODE_DIR / "command.log"


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool
    log_file: Path


class TeeTextStream(io.TextIOBase):
    """Mirror text output to the terminal and an open log file."""

    def __init__(self, terminal: io.TextIOBase, log_handle: io.TextIOBase) -> None:
        self._terminal = terminal
        self._log_handle = log_handle
        self._progress_terminal = self._open_progress_terminal(terminal)

    @property
    def encoding(self) -> str | None:
        return getattr(self._terminal, "encoding", None)

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._terminal.write(text)
        self._log_handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._terminal.fileno()

    def write_progress(self, text: str, *, log_event: bool, final: bool) -> None:
        """Update one terminal row while keeping logs free of control characters."""
        self._progress_terminal.write("\r" + text)
        if final:
            self._progress_terminal.write("\n")
        self._progress_terminal.flush()
        if log_event:
            self._log_handle.write(text + "\n")
            self._log_handle.flush()

    @staticmethod
    def _open_progress_terminal(terminal: io.TextIOBase) -> io.TextIOBase:
        if os.name != "nt" or terminal.isatty():
            return terminal
        try:
            return open("CONOUT$", "w", encoding=getattr(terminal, "encoding", None) or "utf-8")
        except OSError:
            return terminal


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose logging controls on the root parser and every subcommand."""

    group = parser.add_argument_group("logging")
    group.add_argument(
        "--log-file",
        default=argparse.SUPPRESS,
        help="Explicit log file path. Default: Code/command.log in append mode.",
    )
    group.add_argument(
        "--no-log",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable automatic append logging for this run.",
    )
    group.add_argument(
        "--no-progress",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable command progress output for this run.",
    )


def resolve_logging_config(argv: Sequence[str] | None) -> LoggingConfig:
    """Resolve logging settings from raw CLI arguments.

    The parser accepts these flags both before and after the subcommand, so we
    scan the raw argv directly before argparse runs.
    """

    enabled = True
    log_file = DEFAULT_COMMAND_LOG_FILE
    args = list(argv or [])
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--no-log":
            enabled = False
            index += 1
            continue
        if token == "--log-file" and index + 1 < len(args):
            log_file = _resolve_user_path(args[index + 1])
            index += 2
            continue
        if token.startswith("--log-file="):
            log_file = _resolve_user_path(token.split("=", 1)[1])
            index += 1
            continue
        index += 1
    return LoggingConfig(enabled=enabled, log_file=log_file)


@contextmanager
def command_log_context(
    argv: Sequence[str] | None,
    *,
    config: LoggingConfig | None = None,
    program_name: str = "poultry-geo",
) -> Iterator[Path | None]:
    """Capture command stdout and stderr into one append-only project log."""

    settings = config or resolve_logging_config(argv)
    if not settings.enabled:
        yield None
        return

    args = list(argv or [])
    log_path = settings.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    command_text = shlex.join([program_name, *args]).strip()

    with log_path.open("a", encoding="utf-8") as log_handle:
        _write_log_header(
            log_handle,
            log_path=log_path,
            started_at=started_at,
            command_text=command_text or program_name,
        )
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeTextStream(original_stdout, log_handle)
        sys.stderr = TeeTextStream(original_stderr, log_handle)
        try:
            print(f"[logging] Appending command output to {log_path}")
            yield log_path
        except SystemExit as exc:
            exit_code = _normalize_exit_code(exc.code)
            _print_log_footer(
                exit_code=exit_code,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started_perf,
            )
            raise
        except BaseException as exc:
            print("[logging] Command failed with an unhandled exception:", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=log_handle)
            log_handle.flush()
            _print_log_footer(
                exit_code=1,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started_perf,
            )
            raise
        else:
            _print_log_footer(
                exit_code=0,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started_perf,
            )
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _resolve_user_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _normalize_exit_code(raw_code: object) -> int:
    if raw_code is None:
        return 0
    if isinstance(raw_code, bool):
        return int(raw_code)
    if isinstance(raw_code, int):
        return raw_code
    return 1


def _write_log_header(
    log_handle: io.TextIOBase,
    *,
    log_path: Path,
    started_at: datetime,
    command_text: str,
) -> None:
    prefix = "\n" if log_handle.tell() > 0 else ""
    header = [
        prefix + "=" * 80,
        "Command Log",
        "=" * 80,
        f"Started   : {started_at.isoformat()}",
        f"Command   : {command_text}",
        f"CWD       : {Path.cwd()}",
        f"PID       : {os.getpid()}",
        f"Log file  : {log_path}",
        "=" * 80,
        "",
    ]
    log_handle.write("\n".join(header))
    log_handle.flush()


def _print_log_footer(*, exit_code: int, started_at: datetime, elapsed_seconds: float) -> None:
    finished_at = datetime.now().astimezone()
    destination = sys.stderr if exit_code else sys.stdout
    elapsed_text = _format_elapsed(elapsed_seconds)
    print(
        (
            "[logging] Finished "
            f"(exit={exit_code}, started={started_at.isoformat()}, "
            f"finished={finished_at.isoformat()}, duration={elapsed_text}, "
            f"elapsed_seconds={elapsed_seconds:.2f})"
        ),
        file=destination,
    )


def _format_elapsed(elapsed_seconds: float) -> str:
    total_milliseconds = max(0, round(elapsed_seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
