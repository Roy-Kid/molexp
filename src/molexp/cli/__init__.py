"""Command-line interface for molexp.

The ``app`` Typer instance is created here; individual command modules
are imported below for their registration side effects.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)

# Order matters only for --help display. Import for side-effect registration.
from molexp.cli import (  # noqa: E402,F401
    asset,
    experiment,
    project,
    run_cmd,
    runs,
    serve_cmd,
    watch_cmd,
    workspace_cmd,
)

if __name__ == "__main__":
    app()
