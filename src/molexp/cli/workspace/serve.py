"""``molexp serve`` — start FastAPI server + bundled UI.

Serves one *or more* workspaces (local or remote). The first ``--workspace`` is
the active one at startup; the full set is exposed at ``GET /api/workspaces`` so
the UI can switch between them (``POST /api/workspace/open``). With a single
local workspace the behaviour is unchanged (cwd-based).
"""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Annotated

import typer
import uvicorn

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.cli._target import resolve_workspace_target
from molexp.workspace.target import RemoteTarget


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


def _resolve_served(spec: str, used_keys: set[str]):
    """Resolve one ``--workspace`` spec into a ``ServedWorkspace``.

    Remote targets are registered (idempotently) so the active-switch path can
    resolve them by key; local targets get the existing ``workspace.json``
    auto-detection.
    """
    from molexp.server.dependencies import ServedWorkspace, get_workspace_target_registry
    from molexp.server.workspace_targets import WorkspaceTarget

    target, _transport, _fs = resolve_workspace_target(spec)

    if isinstance(target, RemoteTarget):
        host_str = f"{target.user}@{target.host}" if target.user else target.host
        key = _unique_key(_slug(f"{target.host}-{PurePosixPath(target.path).name or 'ws'}"), used_keys)
        registry = get_workspace_target_registry()
        if not registry.has(key):
            registry.add(WorkspaceTarget(name=key, host=host_str, root_path=target.path))
        return ServedWorkspace(key=key, label=str(target), is_remote=True, target_name=key)

    resolved = target.path.resolve()
    if not (resolved / "workspace.json").exists():
        candidate = resolved / "workspace"
        if (candidate / "workspace.json").exists():
            resolved = candidate
            rprint(f"[dim]Auto-detected workspace at {resolved}[/dim]")
        else:
            rprint(
                f"[yellow]Warning:[/yellow] No workspace.json found in {resolved}. "
                "Run [bold]molexp init[/bold] first."
            )
    key = _unique_key(_slug(f"local-{resolved.name or 'ws'}"), used_keys)
    return ServedWorkspace(key=key, label=str(resolved), is_remote=False, path=str(resolved))


@app.command()
def serve(
    workspaces: Annotated[
        list[str] | None,
        typer.Option(
            "--workspace",
            "-ws",
            "--target",
            "-t",
            help=(
                "Workspace to serve: a path, user@host:path, or @target-name. "
                "Repeat to serve several (the first is active at startup). "
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
        set_active_workspace_descriptor,
        set_served_workspaces,
    )

    specs = workspaces or ["."]
    used_keys: set[str] = set()
    served: list[ServedWorkspace] = [_resolve_served(s, used_keys) for s in specs]

    # Activate the first workspace. A local primary stays the cwd (unchanged
    # single-workspace behaviour); a remote primary activates by descriptor.
    primary = served[0]
    if primary.is_remote:
        set_active_workspace_descriptor(primary.target_name)
    else:
        os.chdir(primary.path)  # type: ignore[arg-type]  (path is set for local)
    set_served_workspaces(served)

    if len(served) == 1:
        rprint(f"[bold]Serving Workspace:[/bold] {served[0].label}")
    else:
        rprint(f"[bold]Serving {len(served)} workspaces[/bold] (switch in the UI):")
        for w in served:
            mark = "[green]*[/green]" if w is primary else " "
            kind = "remote" if w.is_remote else "local"
            rprint(f"  {mark} [cyan]{w.key}[/cyan] [dim]({kind})[/dim] {w.label}")

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
