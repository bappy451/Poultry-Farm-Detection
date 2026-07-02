import time
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

progress = Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TextColumn("[green]{task.completed}/{task.total}"),
    "•",
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    "•",
    TextColumn("[yellow]{task.fields[detail]}"),
    expand=True
)

with progress:
    task_id = progress.add_task("spark-grid-train", total=1000, detail="E1 B10 L1:0.1500 Avg:0.1500 LR:1.0e-3")
    for i in range(100):
        time.sleep(0.05)
        progress.update(task_id, advance=10, detail=f"E1 B{10*(i+1)} L1:0.1500 Avg:0.1500 LR:1.0e-3")
