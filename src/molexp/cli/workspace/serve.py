"""``molexp serve`` — start FastAPI server + bundled UI.

Serves one or more local workspaces. The first ``--workspace`` is
the active one at startup; the full set is exposed at ``GET /api/workspaces`` so
the UI can switch between them (``POST /api/workspace/open``). Missing workspace
directories are initialized automatically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.workspace import Workspace


def _slug(text: str) -> str:
    """A filesystem/URL-safe lowercase slug; never empty."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return cleaned or "ws"


def _unique_key(base: str, used: set[str]) -> str:
    """A key derived from *base* that is unique within *used* (mutated)."""
    key, n = base, 2
    while key in used:
        key, n = f"{base}-{n}", n + 1
    used.add(key)
    return key


def _resolve_served(spec: Path, used_keys: set[str]):
    """Resolve one ``--workspace`` spec into a ``ServedWorkspace``.

    The path is expanded and resolved with :mod:`pathlib`. If it is not already
    a MolExp workspace, constructing :class:`Workspace` creates the directory
    and ``workspace.json`` in place.
    """
    from molexp.server.dependencies import ServedWorkspace

    resolved = spec.expanduser().resolve()
    if not (resolved / "workspace.json").exists():
        Workspace(root=resolved, name=resolved.name or "workspace").materialize()
        rprint(f"[dim]Created workspace at {resolved}[/dim]")
    key = _unique_key(_slug(f"local-{resolved.name or 'ws'}"), used_keys)
    return ServedWorkspace(key=key, label=str(resolved), is_remote=False, path=str(resolved))


@app.command()
def serve(
    workspaces: Annotated[
        list[Path] | None,
        typer.Option(
            "--workspace",
            "-ws",
            help=(
                "Local workspace path. Repeat to serve several (the first is active at startup). "
                "Missing workspaces are created automatically. "
                "Default: cwd."
            ),
        ),
    ] = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Server host")] = "localhost",
) -> None:
    """Start the MolExp server (API + bundled web UI)."""
    from molexp.server.dependencies import (
        ServedWorkspace,
        set_served_workspaces,
        set_workspace_path_override,
    )

    specs = workspaces or [Path.cwd()]
    used_keys: set[str] = set()
    served: list[ServedWorkspace] = [_resolve_served(s, used_keys) for s in specs]

    # Activate the first workspace without changing the process cwd.
    primary = served[0]
    assert primary.path is not None
    set_workspace_path_override(Path(primary.path))
    set_served_workspaces(served)

    if len(served) == 1:
        rprint(f"[bold]Serving Workspace:[/bold] {served[0].label}")
    else:
        rprint(f"[bold]Serving {len(served)} workspaces[/bold] (switch in the UI):")
        for w in served:
            mark = "[green]*[/green]" if w is primary else " "
            rprint(f"  {mark} [cyan]{w.key}[/cyan] [dim](local)[/dim] {w.label}")

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
