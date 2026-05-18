"""``molexp workspace serve`` — start FastAPI server + bundled UI."""

from __future__ import annotations

import os
from typing import Annotated

import typer
import uvicorn

from molexp.cli._common import rprint
from molexp.cli.workspace import _get_ctx_target, workspace_app
from molexp.workspace.target import RemoteTarget


@workspace_app.command()
def serve(
    ctx: typer.Context,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8000,
    host: Annotated[str, typer.Option("--host", "-h", help="Server host")] = "localhost",
) -> None:
    """Start the MolExp server (API + bundled web UI)."""
    target = _get_ctx_target(ctx)

    if isinstance(target, RemoteTarget):
        rprint("[red]Error:[/red] Cannot serve a remote workspace.")
        rprint("  Run [bold]molexp serve[/bold] on the remote host, or serve a local workspace.")
        raise typer.Exit(1)

    resolved = target.path.resolve()
    if not (resolved / "workspace.json").exists():
        candidate = resolved / "workspace"
        if (candidate / "workspace.json").exists():
            resolved = candidate
            rprint(f"[dim]Auto-detected workspace at {resolved}[/dim]")
        else:
            rprint(
                f"[yellow]Warning:[/yellow] No workspace.json found in {resolved}. "
                "Run [bold]molexp workspace . init[/bold] first."
            )

    os.chdir(resolved)
    rprint(f"[bold]Serving Workspace:[/bold] {resolved}")

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
