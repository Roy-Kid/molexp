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

``--dev`` additionally spawns the checkout's ``ui/`` Rsbuild server
(``npm run dev``), proxies ``/api`` to this process, and tears the UI down on
Ctrl+C. Open the printed dev-UI URL (not the API port) for HMR.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import uvicorn

from molexp.cli._app import app
from molexp.cli._common import rprint

if TYPE_CHECKING:
    from molexp.server.deps.served import ServedWorkspace
    from molexp.server.workspace_targets import WorkspaceTarget

# Default Rsbuild port for ``--dev`` (docs historically said 5173; rsbuild's
# own default is 3000 — pin so the printed URL is stable).
_DEFAULT_UI_PORT = 5173


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


def _find_ui_dir() -> Path | None:
    """Locate the checkout ``ui/`` directory (source tree only).

    Resolution order:

    1. ``MOLEXP_UI_DIR`` env (explicit override for non-standard layouts)
    2. Walk parents of this module and of the installed ``molexp`` package
       looking for ``ui/package.json`` (editable install from a checkout)
    3. ``None`` when only a wheel is installed (no frontend sources)
    """
    env = os.environ.get("MOLEXP_UI_DIR", "").strip()
    if env:
        candidate = Path(env).expanduser().resolve()
        if (candidate / "package.json").is_file():
            return candidate
        return None

    seeds: list[Path] = [Path(__file__).resolve()]
    try:
        from importlib import resources

        seeds.append(Path(str(resources.files("molexp"))).resolve())
    except Exception:
        pass

    seen: set[Path] = set()
    for seed in seeds:
        for parent in [seed, *seed.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            ui = parent / "ui"
            if (ui / "package.json").is_file():
                return ui
    return None


def _start_ui_dev_server(*, api_port: int, ui_port: int, ui_dir: Path) -> subprocess.Popen[bytes]:
    """Spawn ``npm run dev`` in *ui_dir*, proxying ``/api`` to *api_port*.

    Uses a new process group (POSIX) so Ctrl+C on the parent can SIGTERM the
    whole npm/rsbuild tree without leaving orphan node processes.
    """
    npm = shutil.which("npm")
    if npm is None:
        rprint(
            "[red]Error:[/red] --dev needs npm on PATH "
            "(install Node.js, or unset --dev and use the bundled UI)."
        )
        raise typer.Exit(1)

    # ``--port`` is a real rsbuild CLI flag. The API proxy target must NOT be
    # passed as ``--api-port`` (rsbuild's CAC rejects unknown options) — set
    # ``MOLEXP_API_PORT`` instead (read by ui/rsbuild.config.ts).
    cmd = [
        npm,
        "run",
        "dev",
        "--",
        f"--port={ui_port}",
    ]
    env = os.environ.copy()
    env["MOLEXP_API_PORT"] = str(api_port)
    rprint(f"[dim]Starting UI dev server in {ui_dir}[/dim]")
    rprint(f"[dim]  MOLEXP_API_PORT={api_port} {' '.join(cmd)}[/dim]")

    # start_new_session=True → new process group; killpg on shutdown.
    return subprocess.Popen(
        cmd,
        cwd=ui_dir,
        env=env,
        start_new_session=True,
    )


def _stop_ui_dev_server(proc: subprocess.Popen[bytes] | None, *, timeout: float = 5.0) -> None:
    """Terminate the UI process group started by :func:`_start_ui_dev_server`."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError, PermissionError, OSError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)


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
    port: Annotated[int, typer.Option("--port", "-p", help="API server port")] = 8000,
    host: Annotated[str, typer.Option("--host", help="API server host")] = "localhost",
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help=(
                "Also start the checkout UI dev server (npm run dev) with /api "
                "proxied to this process. Open the printed UI URL for HMR — "
                "not the API port (which still serves the bundled dist if present)."
            ),
        ),
    ] = False,
    ui_port: Annotated[
        int,
        typer.Option(
            "--ui-port",
            help="Rsbuild port when using --dev (default: 5173).",
        ),
    ] = _DEFAULT_UI_PORT,
) -> None:
    """Start the MolExp server (API + bundled web UI).

    With ``--dev``, also starts ``ui/``'s ``npm run dev`` (HMR). Requires a
    source checkout with ``ui/package.json`` and ``npm`` on PATH.
    """
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

    ui_proc: subprocess.Popen[bytes] | None = None
    if dev:
        ui_dir = _find_ui_dir()
        if ui_dir is None:
            rprint(
                "[red]Error:[/red] --dev needs the molexp source tree "
                "(ui/package.json not found). Install from a checkout with "
                "`pip install -e .`, or set MOLEXP_UI_DIR to the ui/ path."
            )
            raise typer.Exit(1)
        ui_proc = _start_ui_dev_server(api_port=port, ui_port=ui_port, ui_dir=ui_dir)
        rprint(f"[cyan]->[/cyan] [bold]Dev UI[/bold]  http://localhost:{ui_port}")
        rprint(f"[cyan]->[/cyan] [dim]API[/dim]     http://{host}:{port}/api")
        webapp = _find_bundled_webapp()
        if webapp is not None:
            rprint(
                f"[dim]  (bundled UI still at http://{host}:{port} — "
                f"use the Dev UI URL above for live reload)[/dim]"
            )
    else:
        webapp = _find_bundled_webapp()
        if webapp is None:
            rprint(f"[cyan]->[/cyan] API at http://{host}:{port}/api  (no bundled UI)")
            rprint(
                "[dim]  Build the frontend (`cd ui && npm run build`), "
                "or use --dev for the HMR UI:[/dim]"
            )
            rprint(f"[dim]  molexp serve --dev -ws … --port {port}[/dim]")
        else:
            rprint(f"[cyan]->[/cyan] http://{host}:{port}")

    application = create_app()
    # SSE streams (approvals / agent tails) hold HTTP connections open. On SIGINT
    # we must wake those generators *before* uvicorn waits for connections to
    # drain — lifespan shutdown runs too late for that. Cap the drain at 3s as
    # a backstop if a client never disconnects.
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=3,
        timeout_keep_alive=5,
    )
    server = uvicorn.Server(config)
    _install_sse_wakeup_on_exit(server, ui_proc=ui_proc)
    try:
        server.run()
    finally:
        _stop_ui_dev_server(ui_proc)


def _install_sse_wakeup_on_exit(
    server: uvicorn.Server,
    *,
    ui_proc: subprocess.Popen[bytes] | None = None,
) -> None:
    """Wrap uvicorn's exit handler so SSE long-polls stop on the first Ctrl+C."""
    original = server.handle_exit

    def handle_exit(sig: int | None, frame: object) -> None:
        try:
            from molexp.server.shutdown import mark_shutting_down
            from molexp.services.approval_notify import close_approval_subscribers

            mark_shutting_down()
            close_approval_subscribers()
        except Exception:  # never block signal handling on a soft failure
            pass
        _stop_ui_dev_server(ui_proc)
        original(sig, frame)

    server.handle_exit = handle_exit  # type: ignore[method-assign]
