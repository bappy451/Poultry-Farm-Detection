from __future__ import annotations

import os
import sys
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence, TextIO


BAR_WIDTH = 32
DEFAULT_REFRESH_SECONDS = 2.0


@dataclass
class ProgressState:
    label: str
    completed: int = 0
    total: int | None = None
    detail: str = ""
    verbose: bool = False


class CommandProgress:
    """Render command progress without claiming a percentage that is not known."""

    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        verbose: bool = False,
    ) -> None:
        self.state = ProgressState(label=label, verbose=verbose)
        self.stream = stream or sys.stdout
        self._live_stream = self._resolve_live_stream(self.stream)
        self.refresh_seconds = refresh_seconds
        self.started_at = time.perf_counter()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pulse = 0
        self._last_width = 0
        self._last_lines = 0

    def start(self) -> None:
        self._write(self._render(), log_event=True)
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def update(
        self,
        completed: int,
        total: int | None = None,
        *,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            self.state.completed = max(0, completed)
            if total is not None:
                self.state.total = max(0, total)
            if detail is not None:
                self.state.detail = detail
            completed_all = (
                self.state.total is not None
                and self.state.total > 0
                and self.state.completed >= self.state.total
            )
        self._write(self._render(), log_event=True, final=completed_all)

    def finish(self, *, success: bool) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.refresh_seconds * 2))
        with self._lock:
            already_complete = (
                success
                and self.state.total is not None
                and self.state.total > 0
                and self.state.completed >= self.state.total
            )
        if already_complete:
            return
        elapsed = time.perf_counter() - self.started_at
        if success:
            bar = "=" * BAR_WIDTH
            status = "100%"
        else:
            bar = "!" * BAR_WIDTH
            status = "FAILED"
        self._write(
            f"{self.state.label:<20} [{bar}] {status:>6} | elapsed {_format_elapsed(elapsed)}",
            log_event=True,
            final=True,
        )

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self.refresh_seconds):
            self._pulse += 1
            self._write(self._render(), log_event=False)

    def _render(self) -> str:
        with self._lock:
            state = ProgressState(**vars(self.state))
        elapsed = time.perf_counter() - self.started_at
        if state.total is not None and state.total > 0:
            ratio = min(1.0, state.completed / state.total)
            filled = round(ratio * BAR_WIDTH)
            bar = "=" * filled + " " * (BAR_WIDTH - filled)
            percent = f"{ratio * 100:3.0f}%"
            count = f" | {state.completed}/{state.total}"
            if state.completed > 0 and ratio < 1.0:
                eta_seconds = elapsed * (state.total - state.completed) / state.completed
                eta = f" | ETA {_format_elapsed(eta_seconds)}"
            else:
                eta = " | ETA 00:00:00"
        else:
            position = self._pulse % BAR_WIDTH
            bar_chars = [" "] * BAR_WIDTH
            bar_chars[position] = ">"
            bar = "".join(bar_chars)
            percent = "working"
            count = ""
            eta = " | ETA --"
        
        main_line = (
            f"{state.label:<20} [{bar}] {percent:>7}"
            f" | elapsed {_format_elapsed(elapsed)}{eta}{count}"
        )

        if state.verbose and state.detail:
            # If verbose, we might want to split detail into multiple lines or show it prominently
            # For now, let's keep it simple but ensure it's not truncated if we can help it.
            return f"{main_line}\n  > {state.detail}"
        
        detail = f" | {state.detail}" if state.detail else ""
        return f"{main_line}{detail}"

    def _write(self, content: str, *, log_event: bool, final: bool = False) -> None:
        lines = content.split("\n")
        rendered_lines = [self._fit_terminal(f"[progress] {line}") for line in lines]
        
        # Move cursor up if we wrote multiple lines last time
        if self._last_lines > 1:
            self._live_stream.write(f"\x1b[{self._last_lines - 1}A")
        
        # Clear lines and write new content
        for i, line in enumerate(rendered_lines):
            self._live_stream.write("\r\x1b[K" + line)
            if i < len(rendered_lines) - 1:
                self._live_stream.write("\n")
        
        if final:
            self._live_stream.write("\n")
            self._last_lines = 0
        else:
            self._live_stream.write("\r")
            self._last_lines = len(rendered_lines)
            
        self._live_stream.flush()
        
        if log_event and self._live_stream is not self.stream:
            self._append_wrapper_log("\n".join(rendered_lines))

    @staticmethod
    def _fit_terminal(line: str) -> str:
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        usable_width = max(40, width - 1)
        if len(line) <= usable_width:
            return line
        return line[: usable_width - 3] + "..."

    @staticmethod
    def _resolve_live_stream(stream: TextIO) -> TextIO:
        if stream not in {sys.stdout, sys.stderr}:
            return stream
        if os.name != "nt" or stream.isatty():
            return stream
        try:
            return open("CONOUT$", "w", encoding=getattr(stream, "encoding", None) or "utf-8")
        except OSError:
            return stream

    @staticmethod
    def _append_wrapper_log(line: str) -> None:
        log_path = os.environ.get("POULTRY_PROGRESS_LOG")
        if not log_path:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            return


@contextmanager
def command_progress_context(
    argv: Sequence[str] | None,
    *,
    stream: TextIO | None = None,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
) -> Iterator[CommandProgress | None]:
    args = list(argv or [])
    if "--no-progress" in args or any(token in {"-h", "--help", "-?"} for token in args):
        yield None
        return

    progress = CommandProgress(
        _command_label(args),
        stream=stream,
        refresh_seconds=refresh_seconds,
        verbose="--verbose" in args,
    )
    progress.start()
    try:
        yield progress
    except BaseException:
        progress.finish(success=False)
        raise
    else:
        progress.finish(success=True)


def _command_label(args: Sequence[str]) -> str:
    options_with_values = {"--log-file"}
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return "poultry-geo"


def _format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = max(0, round(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
