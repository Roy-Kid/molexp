"""``molexp serve`` — start the FastAPI server + bundled UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from . import app
from ._common import rprint


@app.command()
def serve(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace root (default: current directory).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = Path.cwd(),
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port"),
    ] = 8000,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Server host"),
    ] = "localhost",
) -> None:
    """Start the MolExp server (API + bundled web UI)."""
    resolved = Path(workspace).resolve()
    if not (resolved / "workspace.json").exists():
        candidate = resolved / "workspace"
        if (candidate / "workspace.json").exists():
            resolved = candidate
            rprint(f"[dim]Auto-detected workspace at {resolved}[/dim]")
        else:
            rprint(
                f"[yellow]Warning:[/yellow] No workspace.json found in {resolved}. "
                "Run [bold]molexp init[/bold] or use [bold]--workspace[/bold]."
            )

    os.chdir(resolved)
    rprint(f"[bold]Serving Workspace:[/bold] {workspace}")

    from molexp.server.app import _find_bundled_webapp, create_app

    webapp = _find_bundled_webapp()
    if webapp is None:
        rprint(f"[cyan]->[/cyan] API at http://{host}:{port}/api  (no bundled UI)")
        rprint(
            "[dim]  Build a wheel to include the frontend, "
            "or run the frontend dev server separately:[/dim]"
        )
        rprint(f"[dim]  cd ui && npm run dev -- --api-port={port}[/dim]")
    else:
        rprint(f"[cyan]->[/cyan] http://{host}:{port}")

    application = create_app()
    uvicorn.run(application, host=host, port=port, log_level="info")
