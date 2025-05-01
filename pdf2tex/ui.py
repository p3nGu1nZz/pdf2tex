"""UI elements like console and progress bar configuration."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.theme import Theme

# Define a custom theme (dark with purple accents)
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "status": "bold bright_magenta",  # Purple-like for status
    "progress.description": "white",
    "progress.percentage": "bright_magenta",
    "bar.complete": "bright_magenta",
    "bar.finished": "green",
    "bar.pulse": "bright_magenta",
    "progress.remaining": "dim cyan",
    "progress.elapsed": "dim white",
})
console = Console(theme=custom_theme)

# Configure Progress columns
progress_columns = [
    TextColumn("[progress.description]{task.description}", justify="right"),
    BarColumn(
        bar_width=None,
        complete_style="bar.complete",
        finished_style="bar.finished",
        pulse_style="bar.pulse"
    ),
    "[progress.percentage]{task.percentage:>3.1f}%",
    TimeRemainingColumn(),
    TimeElapsedColumn(),
]