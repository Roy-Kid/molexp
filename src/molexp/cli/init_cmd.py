"""``molexp init`` — ergonomic top-level shortcut for workspace initialization.

The canonical command is ``molexp workspace <TARGET> init``, but the
single-arg ``molexp init [PATH]`` form is so common we expose it at the
top level too.  Both paths converge on ``Workspace(...).materialize()``.

Behavior:
- ``molexp init <path>`` — create or refresh the workspace at *path*
- ``molexp init`` — same, on the current working directory
- Idempotent: re-running on an existing workspace leaves child state
  (e.g. ``projects/``) intact and only refreshes ``workspace.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from molexp.cli._common import rprint


def init(
    path: Annotated[
        Path | None,
        typer.Argument(help="Workspace path (defaults to current directory)"),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Workspace name (derived from path if omitted)"),
    ] = None,
) -> None:
    """Initialize (or refresh) a workspace at PATH (defaults to current dir)."""
    from molexp.workspace import Workspace

    target = Path(path) if path is not None else Path.cwd()
    rprint(f"[bold]Initializing workspace at:[/bold] {target}")
    try:
        ws = Workspace(str(target), name=name)
        ws.materialize()
    except Exception as exc:
        rprint(f"[red]Failed to initialize workspace:[/red] {exc}")
        raise typer.Exit(1) from exc

    rprint(f"[green]OK[/green] Workspace ready: {ws.root}")
