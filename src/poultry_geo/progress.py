from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import TextIO

from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn,
)


class CommandProgress:
    def __init__(self, label: str, verbose: bool = False) -> None:
        self.label = label
        self.verbose = verbose
        
        import shutil
        
        # Bypass PowerShell Tee-Object piping by drawing directly to the active console buffer.
        stream = sys.stdout
        force_term = None
        if os.name == "nt" and not stream.isatty():
            try:
                stream = open("CONOUT$", "w", encoding="utf-8")
                force_term = True
            except OSError:
                pass
                
        # When run inside a PowerShell subprocess, Python often falsely detects the terminal
        # width as 80 columns, causing Rich to aggressively truncate text into '...'.
        # We forcefully override this to a wide layout so your text never truncates.
        term_width = shutil.get_terminal_size(fallback=(160, 24)).columns
        if term_width < 140:
            term_width = 160
            
        console = Console(file=stream, force_terminal=force_term, width=term_width)

        # Create a highly modern, premium, and beautiful progress bar
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bold bright_green"),
            TextColumn("[bold bright_cyan]{task.description}"),
            BarColumn(bar_width=20, complete_style="bright_green", finished_style="bold bright_green", pulse_style="bright_white"),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TextColumn("[dim]Elapsed:[/dim]"),
            TimeElapsedColumn(),
            "•",
            TextColumn("[dim]ETA:[/dim]"),
            TimeRemainingColumn(),
            "•",
            TextColumn("[bold bright_yellow]{task.fields[detail]}"),
            console=console,
            transient=False,
            refresh_per_second=10,
        )
        self.task_id = None
        self._log_path = os.environ.get("POULTRY_PROGRESS_LOG")

    def start(self) -> None:
        self.progress.start()

    def update(
        self,
        completed: int,
        total: int | None = None,
        *,
        detail: str | None = None,
        description: str | None = None,
    ) -> None:
        safe_detail = detail or ""
        
        if self.task_id is None:
            # Initialize the task on first update
            self.task_id = self.progress.add_task(
                description or self.label, 
                total=total or 100, 
                detail=safe_detail
            )
            
        if total is not None:
            self.progress.update(self.task_id, total=total)
            
        if description is not None:
            self.progress.update(self.task_id, description=description)
            
        self.progress.update(self.task_id, completed=completed, detail=safe_detail)
        
        # Optionally log progress to a file if requested by the environment
        if self._log_path and completed % 10 == 0:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{self.label}] {completed}/{total} | {safe_detail}\n")
            except OSError:
                pass

    def finish(self, *, success: bool) -> None:
        if self.task_id is not None:
            if success:
                total = self.progress.tasks[self.task_id].total
                if total:
                    self.progress.update(self.task_id, completed=total, detail="[bold green]Completed")
            else:
                self.progress.update(self.task_id, detail="[bold red]Failed")
                
        self.progress.stop()


@contextmanager
def command_progress_context(
    argv: Sequence[str] | None,
    *,
    stream: TextIO | None = None,
    refresh_seconds: float = 2.0,
) -> Iterator[CommandProgress | None]:
    args = list(argv or [])
    if "--no-progress" in args or any(token in {"-h", "--help", "-?"} for token in args):
        yield None
        return

    progress = CommandProgress(
        _command_label(args),
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
    options_with_values = {"--log-file", "--config"}
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
