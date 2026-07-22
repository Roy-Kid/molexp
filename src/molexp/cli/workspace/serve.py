"""``molexp serve`` — start FastAPI server + bundled UI.

Serves one or more roots. The first ``--workspace`` is active at startup; the
full set is exposed at ``GET /api/workspaces`` so the UI can switch between
them (``POST /api/workspace/open``).

A root may be:

* a local path (any directory — plain folders open without writing
  ``workspace.json``; missing dirs are created empty)
* an SCP remote ``user@host:/path`` (lazy-download mirror via
  :class:`~molexp.workspace.fs_cached.CachedRemoteFileSystem`)
* a registered workspace target ``@name`` (from
  ``~/.molexp/workspace_targets.json``)

MolExp project/run indexes require a full workspace layout; plain folders
still serve file-tree / content routes with empty index views.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import uvicorn

from molexp.cli._app import app
from molexp.cli._common import rprint

if TYPE_CHECKING:
    from molexp.server.deps.served import ServedWorkspace
    from molexp.server.workspace_targets import WorkspaceTarget


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


def _local_served(resolved: Path, used_keys: set[str]) -> ServedWorkspace:
    """Build a local :class:`ServedWorkspace` for an existing-or-created dir."""
    from molexp.server.dependencies import ServedWorkspace

    if resolved.exists() and not resolved.is_dir():
        rprint(f"[red]Error:[/red] not a directory: {resolved}")
        raise typer.Exit(1)
    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
        rprint(f"[dim]Created directory at {resolved}[/dim]")
    # Do NOT materialize workspace.json — plain folders are valid roots.
    key = _unique_key(_slug(f"local-{resolved.name or 'ws'}"), used_keys)
    return ServedWorkspace(key=key, label=str(resolved), is_remote=False, path=str(resolved))


def _remote_served(
    *,
    name: str,
    label: str,
    used_keys: set[str],
    remote_target: WorkspaceTarget,
) -> ServedWorkspace:
    """Build a remote :class:`ServedWorkspace` with an inline target descriptor."""
    from molexp.server.dependencies import ServedWorkspace

    key = _unique_key(_slug(f"remote-{name}"), used_keys)
    return ServedWorkspace(
        key=key,
        label=label,
        is_remote=True,
        target_name=name,
        remote_target=remote_target,
    )


def _resolve_served(spec: str | Path, used_keys: set[str]) -> ServedWorkspace:
    """Resolve one ``--workspace`` spec into a :class:`ServedWorkspace`.

    Accepts local paths, SCP ``user@host:/path``, and registered
    ``@workspace-target`` names. Never writes ``workspace.json`` for local
    roots; missing local directories are created empty.
    """
    from molexp.server.deps.targets import get_workspace_target_registry
    from molexp.server.workspace_targets import WorkspaceTarget
    from molexp.workspace.target import (
        LocalTarget,
        RemoteTarget,
        TargetNeedsResolution,
        parse_target,
    )

    raw = str(spec).strip() or "."

    # ``@name`` → registered WorkspaceTarget (server registry, not compute targets).
    if raw.startswith("@"):
        name = raw[1:]
        if not name:
            rprint("[red]Error:[/red] empty workspace target name after @")
            raise typer.Exit(1)
        registry = get_workspace_target_registry()
        try:
            target = registry.get(name)
        except KeyError as exc:
            rprint(
                f"[red]Error:[/red] no workspace target named {name!r} "
                f"(register via Settings / POST /api/workspace/targets)"
            )
            raise typer.Exit(1) from exc
        return _remote_served(
            name=target.name,
            label=f"{target.host}:{target.root_path}",
            used_keys=used_keys,
            remote_target=target,
        )

    try:
        parsed = parse_target(raw)
    except TargetNeedsResolution as exc:
        # parse_target only raises this for ``@name`` — handled above.
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if isinstance(parsed, LocalTarget):
        resolved = Path(parsed.path).expanduser().resolve()
        return _local_served(resolved, used_keys)

    assert isinstance(parsed, RemoteTarget)
    host_part = f"{parsed.user}@{parsed.host}" if parsed.user else (parsed.host or "remote")
    root = parsed.path
    # target_name is process-stable for the active-workspace descriptor; the
    # ServedWorkspace.key is separately uniquified by _remote_served.
    target_name = _slug(f"{host_part}-{Path(root).name or 'ws'}")
    wt = WorkspaceTarget(
        name=target_name,
        host=host_part,
        port=parsed.port,
        identity_file=parsed.identity_file,
        ssh_opts=tuple(parsed.ssh_opts) if parsed.ssh_opts else (),
        root_path=root,
    )
    return _remote_served(
        name=target_name,
        label=str(parsed),
        used_keys=used_keys,
        remote_target=wt,
    )


@app.command()
def serve(
    workspaces: Annotated[
        list[str] | None,
        typer.Option(
            "--workspace",
            "-ws",
            help=(
                "Root to serve: local path, user@host:/path, or @workspace-target. "
                "Repeat for several (first is active at startup). "
                "Plain folders open without writing workspace.json; missing "
                "local dirs are created empty. Default: cwd."
            ),
        ),
    ] = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Server host")] = "localhost",
) -> None:
    """Start the MolExp server (API + bundled web UI)."""
    from molexp.server.dependencies import (
        set_active_workspace_descriptor,
        set_served_workspaces,
        set_workspace_path_override,
    )

    specs: list[str] = list(workspaces) if workspaces else [str(Path.cwd())]
    used_keys: set[str] = set()
    served = [_resolve_served(s, used_keys) for s in specs]

    # Activate the first root without changing the process cwd.
    primary = served[0]
    if primary.is_remote:
        assert primary.target_name is not None
        set_active_workspace_descriptor(primary.target_name)
    else:
        assert primary.path is not None
        set_workspace_path_override(Path(primary.path))
    set_served_workspaces(served)

    if len(served) == 1:
        kind = "remote" if served[0].is_remote else "local"
        rprint(f"[bold]Serving root:[/bold] {served[0].label} [dim]({kind})[/dim]")
    else:
        rprint(f"[bold]Serving {len(served)} roots[/bold] (switch in the UI):")
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
