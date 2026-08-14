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

``--dev`` additionally spawns the checkout's ``apps/web`` leaf script
``npm run dev:api`` (Rsbuild HMR against this process's API — not the MSW mock
``npm run dev`` / root ``npm run dev:web``), and tears the UI down on Ctrl+C.
Open the printed dev-UI URL (not the API port) for HMR.

``--tunnel`` punches a public HTTPS hole (cloudflared or zrok) while the
API stays on localhost. With ``--dev`` the hole waits for Rsbuild's
``Local:`` URL, then dials whatever loopback actually accepted that port
(so cloudflared tracks the Dev UI; Rsbuild is not reconfigured). Without
``--dev`` it targets the API port (bundled UI). Provider and tokens come
from ``--via`` / ``--tunnel-token`` or ``molexp config`` (``tunnel.*``) —
never env vars. Ctrl+C stops tunnel + server.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlparse

import typer
import uvicorn

from molexp._logger import get_logger
from molexp.cli._app import app
from molexp.cli._common import rprint

_log = get_logger("molexp.tunnel")

if TYPE_CHECKING:
    from types import FrameType

    from molexp.server.deps.served import ServedWorkspace
    from molexp.server.tunnel.base import TunnelBackend
    from molexp.server.workspace_targets import WorkspaceTarget

# Default Rsbuild port for ``--dev`` (docs historically said 5173; rsbuild's
# own default is 3000 — pin so the printed URL is stable).
_DEFAULT_UI_PORT = 5173
_DEV_UI_LOCAL_RE = re.compile(r"Local:\s+(https?://\S+)", re.IGNORECASE)


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
    # Soft preflight for 2FA hosts: if ControlMaster is down, BatchMode
    # serve calls cannot answer an OTP prompt.  Key-only hosts establish a
    # master on first op automatically — the check is advisory only.
    try:
        from molq.options import SshTransportOptions
        from molq.transport import SshTransport

        ssh = SshTransport(
            options=SshTransportOptions(
                host=parsed.host or host_part,
                port=parsed.port,
                identity_file=parsed.identity_file,
                ssh_opts=tuple(parsed.ssh_opts) if parsed.ssh_opts else (),
            )
        )
        is_alive = getattr(ssh, "is_master_alive", None)
        if callable(is_alive) and not is_alive():
            rprint(
                f"[dim]Tip:[/dim] no SSH ControlMaster for [bold]{host_part}[/bold]. "
                f"If this host needs a verification code, run "
                f"[bold]molexp connect -ws {raw}[/bold] first."
            )
    except Exception:
        pass

    return _remote_served(
        name=target_name,
        label=str(parsed),
        used_keys=used_keys,
        remote_target=wt,
    )


def _find_web_dir() -> Path | None:
    """Locate the checkout ``apps/web`` directory (source tree only).

    Resolution order:

    1. ``MOLEXP_WEB_DIR`` env (explicit override for non-standard layouts)
    2. Walk parents of this module and of the installed ``molexp`` package
       looking for ``apps/web/package.json`` (editable install from a checkout)
    3. ``None`` when only a wheel is installed (no frontend sources)
    """
    env = os.environ.get("MOLEXP_WEB_DIR", "").strip()
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
            web = parent / "apps" / "web"
            if (web / "package.json").is_file():
                return web
    return None


def _start_web_dev_server(
    *,
    api_port: int,
    web_port: int,
    web_dir: Path,
    capture: bool = False,
) -> subprocess.Popen[bytes]:
    """Spawn the ``apps/web`` leaf ``npm run dev:api`` in *web_dir*, proxying ``/api``.

    Uses a new process group (POSIX) so Ctrl+C on the parent can SIGTERM the
    whole npm/rsbuild tree without leaving orphan node processes. Repo-root
    equivalent: ``npm run dev:api`` (not ``dev:web``, which is the MSW mock).

    *capture* pipes stdout/stderr so the caller can read Rsbuild's ``Local:``
    line (used by ``--dev --tunnel``). Lines are still echoed to this process.
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
    # ``MOLEXP_API_PORT`` instead (read by apps/web/rsbuild.config.ts).
    # Leaf ``dev:api`` = real proxy; leaf ``dev`` / root ``dev:web`` = MSW mock.
    cmd = [
        npm,
        "run",
        "dev:api",
        "--",
        f"--port={web_port}",
    ]
    env = os.environ.copy()
    env["MOLEXP_API_PORT"] = str(api_port)
    rprint(f"[dim]Starting web dev server in {web_dir}[/dim]")
    rprint(f"[dim]  MOLEXP_API_PORT={api_port} {' '.join(cmd)}[/dim]")

    # start_new_session=True → new process group; killpg on shutdown.
    if capture:
        return subprocess.Popen(
            cmd,
            cwd=web_dir,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return subprocess.Popen(
        cmd,
        cwd=web_dir,
        env=env,
        start_new_session=True,
    )


def _stop_web_dev_server(proc: subprocess.Popen[bytes] | None, *, timeout: float = 5.0) -> None:
    """Terminate the web process group started by :func:`_start_web_dev_server`."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
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


def tunnel_local_port(*, dev: bool, api_port: int, ui_port: int) -> int:
    """Local port the public tunnel should punch.

    ``--dev --tunnel`` forwards the Rsbuild Dev UI (HMR + ``/api`` proxy).
    Without ``--dev`` the API process itself serves the bundled UI.
    """
    return ui_port if dev else api_port


def tunnel_local_host(*, dev: bool, api_host: str, ui_host: str) -> str:
    """Local host the public tunnel should dial.

    ``--dev`` uses the discovered Dev UI bind. Without ``--dev`` the API
    bind is used (loopback coerced to 127.0.0.1).
    """
    if dev:
        return ui_host
    if api_host in ("localhost", "127.0.0.1", "::1", "[::1]"):
        return "127.0.0.1"
    return api_host


def parse_dev_ui_url(line: str) -> str | None:
    """Extract the Rsbuild ``Local:`` URL from one log line, if present."""
    m = _DEV_UI_LOCAL_RE.search(line)
    if m is None:
        return None
    return m.group(1).rstrip("/").rstrip(".,);]")


def origin_from_dev_ui_url(url: str) -> tuple[str, int]:
    """Host and port from a Dev UI origin URL (IPv6 hostname is unbracketed)."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _tcp_open(host: str, port: int, *, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def dev_ui_listen_host(
    port: int,
    *,
    probe: Callable[[str, int], bool] | None = None,
    timeout: float = 5.0,
) -> str | None:
    """First loopback family that accepts *port* (IPv4, then IPv6)."""
    check = probe or _tcp_open
    deadline = time.monotonic() + timeout
    while True:
        for host in ("127.0.0.1", "::1"):
            if check(host, port):
                return host
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _await_dev_ui_origin(
    proc: subprocess.Popen[bytes],
    *,
    fallback_port: int,
    timeout: float = 45.0,
    probe: Callable[[str, int], bool] | None = None,
) -> tuple[str, str, int]:
    """Read Rsbuild ``Local:``, then pick a dialable loopback.

    Returns ``(advertised_host, dial_host, port)``. *advertised_host* is
    what Rsbuild printed (usually ``localhost``); *dial_host* is the
    family that actually accepted the port (``127.0.0.1`` or ``::1``).
    """
    advertised_port = fallback_port
    advertised_host = "localhost"
    seen = threading.Event()

    def _read() -> None:
        nonlocal advertised_host, advertised_port
        stdout = proc.stdout
        if stdout is None:
            return
        for raw in stdout:
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            sys.stdout.write(text)
            sys.stdout.flush()
            url = parse_dev_ui_url(text)
            if url is None:
                continue
            advertised_host, advertised_port = origin_from_dev_ui_url(url)
            seen.set()

    if proc.stdout is not None:
        threading.Thread(target=_read, name="dev-ui-log", daemon=True).start()
        seen.wait(timeout=timeout)

    dial = dev_ui_listen_host(advertised_port, probe=probe, timeout=min(8.0, timeout))
    if dial is None:
        dial = advertised_host
    return advertised_host, dial, advertised_port


def access_banner_lines(
    *,
    dev: bool,
    host: str,
    api_port: int,
    ui_port: int,
    public_url: str | None = None,
    ui_host: str = "localhost",
) -> list[str]:
    """Rich markup lines naming Dev UI / API origins for the serve banner."""
    origin = public_url.rstrip("/") if public_url else None
    local_ui = f"http://{ui_host}:{ui_port}"
    local_api = f"http://{host}:{api_port}/api"
    if dev:
        if origin is None:
            return [
                f"[cyan]->[/cyan] [bold]Dev UI[/bold]  {local_ui}",
                f"[cyan]->[/cyan] [dim]API[/dim]     {local_api}",
            ]
        return [
            f"[cyan]->[/cyan] [bold]Dev UI[/bold]  {origin}",
            f"           {local_ui}",
            f"[cyan]->[/cyan] [dim]API[/dim]     {origin}/api",
            f"           {local_api}",
        ]
    if origin is None:
        return [f"[cyan]->[/cyan] [dim]API[/dim]     {local_api}"]
    return [
        f"[cyan]->[/cyan] [dim]API[/dim]     {origin}",
        f"           {origin}/api",
        f"           {local_api}",
    ]


def _print_access_banner(
    *,
    dev: bool,
    host: str,
    api_port: int,
    ui_port: int,
    public_url: str | None = None,
    ui_host: str = "localhost",
) -> None:
    for line in access_banner_lines(
        dev=dev,
        host=host,
        api_port=api_port,
        ui_port=ui_port,
        public_url=public_url,
        ui_host=ui_host,
    ):
        rprint(line)


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
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            help=(
                "API server host (default: localhost; with --tunnel defaults to "
                "127.0.0.1 so only the tunnel is public)."
            ),
        ),
    ] = None,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help=(
                "Also start the checkout web UI against this API (apps/web leaf: "
                "npm run dev:api; repo root: npm run dev:api) with /api proxied "
                "to this process. Open the printed UI URL for HMR — not the API "
                "port (which still serves the bundled dist if present). "
                "For offline MSW mock UI use root npm run dev:web instead."
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
    auth: Annotated[
        bool,
        typer.Option(
            "--auth",
            help=(
                "Require login for the HTTP API + UI (filesystem users under "
                "~/.molexp/auth/). Also on when auth.enabled is true in "
                "~/.molexp/config.json. Non-loopback --host refuses to start "
                "without auth."
            ),
        ),
    ] = False,
    user: Annotated[
        str | None,
        typer.Option(
            "--user",
            "-u",
            help=(
                "When --auth is on, require this username to already exist "
                "(default check skipped unless set). Bootstrap with "
                "`molexp auth login -u admin`."
            ),
        ),
    ] = None,
    tunnel: Annotated[
        bool,
        typer.Option(
            "--tunnel/--no-tunnel",
            help=(
                "Punch a public HTTPS hole to this serve. Provider comes from "
                "--via or `molexp config` tunnel.via (default: cloudflared)."
            ),
        ),
    ] = False,
    via: Annotated[
        str | None,
        typer.Option(
            "--via",
            help="Tunnel provider: cloudflared or zrok. Overrides tunnel.via.",
        ),
    ] = None,
    tunnel_mode: Annotated[
        str | None,
        typer.Option(
            "--tunnel-mode",
            help=("cloudflared: quick|named. zrok: public|reserved. Overrides tunnel.mode."),
        ),
    ] = None,
    tunnel_hostname: Annotated[
        str | None,
        typer.Option(
            "--tunnel-hostname",
            help="Public hostname for named cloudflared tunnels. Overrides tunnel.hostname.",
        ),
    ] = None,
    tunnel_token: Annotated[
        str | None,
        typer.Option(
            "--tunnel-token",
            help=(
                "Named cloudflared token, or reserved zrok share token. "
                "Overrides tunnel.token (not read from the environment)."
            ),
        ),
    ] = None,
    tunnel_bin: Annotated[
        str | None,
        typer.Option(
            "--tunnel-bin",
            help="Path to cloudflared/zrok. Overrides tunnel.bin (else PATH).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="If a tunnel client is missing, download it to ~/.local/bin without asking.",
        ),
    ] = False,
) -> None:
    """Start the MolExp server (API + bundled web UI).

    With ``--dev``, also starts the ``apps/web`` leaf ``npm run dev:api`` (HMR
    against this API; root equivalent ``npm run dev:api``). Requires a source
    checkout with ``apps/web/package.json`` and ``npm`` on PATH. Offline mock
    UI is ``npm run dev:web``, not this path.

    With ``--tunnel``, punches a public HTTPS hole and prints the URL.
    """
    from molexp.cli.workspace._serve_auth import configure_serve_auth
    from molexp.server.dependencies import (
        set_active_workspace_descriptor,
        set_served_workspaces,
        set_workspace_path_override,
    )

    use_tunnel = tunnel
    # Default bind: loopback; tunnel forces 127.0.0.1 unless user set --host.
    if host is None:
        host = "127.0.0.1" if use_tunnel else "localhost"
    elif use_tunnel and host not in ("127.0.0.1", "localhost", "::1"):
        rprint(
            f"[yellow]Warning:[/yellow] --tunnel with --host {host} exposes the "
            f"API beyond loopback; prefer 127.0.0.1 and the tunnel URL only."
        )
    configure_serve_auth(host=host, auth_flag=auth, require_user=user)

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

    web_proc: subprocess.Popen[bytes] | None = None
    tunnel_handle: TunnelBackend | None = None
    advertised_ui_host = "localhost"
    dial_ui_host = "localhost"
    dial_ui_port = ui_port
    if dev:
        web_dir = _find_web_dir()
        if web_dir is None:
            rprint(
                "[red]Error:[/red] --dev needs the molexp source tree "
                "(apps/web/package.json not found). Install from a checkout "
                "with `pip install -e .`, or set MOLEXP_WEB_DIR to the "
                "apps/web path."
            )
            raise typer.Exit(1)
        web_proc = _start_web_dev_server(
            api_port=port,
            web_port=ui_port,
            web_dir=web_dir,
            capture=use_tunnel,
        )
        if use_tunnel:
            advertised_ui_host, dial_ui_host, dial_ui_port = _await_dev_ui_origin(
                web_proc, fallback_port=ui_port
            )
        webapp = _find_bundled_webapp()
        if not use_tunnel:
            _print_access_banner(
                dev=True,
                host=host,
                api_port=port,
                ui_port=ui_port,
                ui_host=advertised_ui_host,
            )
            if webapp is not None:
                rprint(
                    f"[dim]  (bundled UI still at http://{host}:{port} — "
                    f"use the Dev UI URL above for live reload)[/dim]"
                )
    else:
        webapp = _find_bundled_webapp()
        if webapp is None:
            _log.warning(
                "API-only: no bundled UI at src/molexp/dist. "
                "Build with `npm run build:web`, or `molexp serve --dev`."
            )
        else:
            _log.info("local UI", url=f"http://{host}:{port}")

    if use_tunnel:
        from dataclasses import replace

        from molexp.server.tunnel import (
            TunnelError,
            ensure_tunnel_client,
            open_tunnel,
            resolve_tunnel_settings,
        )

        try:
            settings = resolve_tunnel_settings(
                via=via,
                mode=tunnel_mode,
                token=tunnel_token,
                bin=tunnel_bin,
                hostname=tunnel_hostname,
            )
            _log.info("looking for tunnel client", via=settings.via)
            settings = replace(
                settings,
                bin=ensure_tunnel_client(
                    via=settings.via,
                    explicit=settings.bin,
                    ask=(
                        (lambda binary, default_dir: default_dir / binary)
                        if yes
                        else _ask_tunnel_download
                    ),
                ),
            )
            _log.info("using tunnel client", path=settings.bin, via=settings.via)
            _log.info("starting tunnel", via=settings.via, mode=settings.mode)
            tunnel_handle = open_tunnel(
                local_port=tunnel_local_port(dev=dev, api_port=port, ui_port=dial_ui_port),
                settings=settings,
                local_host=tunnel_local_host(dev=dev, api_host=host, ui_host=dial_ui_host),
            )
            tunnel_handle.start()
        except TunnelError as exc:
            _log.error(str(exc))
            _stop_web_dev_server(web_proc)
            raise typer.Exit(1) from exc
        # Print once after wait — on_url runs on the reader thread.
        try:
            url = tunnel_handle.wait_for_url()
        except TunnelError as exc:
            _log.error(str(exc))
            _stop_tunnel(tunnel_handle)
            _stop_web_dev_server(web_proc)
            raise typer.Exit(1) from exc
        if url is None:
            rprint("[yellow]Warning:[/yellow] public URL not detected yet; watch client output")
            _print_access_banner(
                dev=dev,
                host=host,
                api_port=port,
                ui_port=dial_ui_port,
                ui_host=advertised_ui_host,
            )
        else:
            _print_access_banner(
                dev=dev,
                host=host,
                api_port=port,
                ui_port=dial_ui_port,
                public_url=url,
                ui_host=advertised_ui_host,
            )
        if dev and webapp is not None:
            rprint(
                f"[dim]  (bundled UI still at http://{host}:{port} — "
                f"use the Dev UI URL above for live reload)[/dim]"
            )
        _log.info("anyone with the url can reach this workspace — stop with Ctrl+C")

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
    _install_sse_wakeup_on_exit(server, web_proc=web_proc, tunnel=tunnel_handle)
    try:
        server.run()
    finally:
        _stop_tunnel(tunnel_handle)
        _stop_web_dev_server(web_proc)


def _ask_tunnel_download(binary: str, default_dir: Path) -> Path | None:
    """Ask where to put a missing tunnel client. Non-TTY → skip.

    Reply: ``y`` default dir, ``n`` cancel, or a custom directory/file path.
    """
    from molexp.server.tunnel.fetch import interpret_download_reply

    if not sys.stdin.isatty():
        rprint(
            f"[yellow]{binary} not found on PATH; not a TTY, skipping download "
            f"(re-run with -y to fetch to {default_dir}/).[/yellow]"
        )
        return None
    # Print first and flush — a bare typer.prompt after Rich output is easy to
    # miss and looks like the process hung.
    rprint(f"[yellow]{binary} not found on PATH.[/yellow]")
    rprint(
        f"Download the official {binary} to [bold]{default_dir}/[/bold]? "
        "[dim]y = yes · n = cancel · or type a path[/dim]"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    reply = typer.prompt("[y/N/path]", default="N", show_default=False)
    dest = interpret_download_reply(reply, binary=binary, default_dir=default_dir)
    if dest is not None:
        _log.info("fetching tunnel client", dest=str(dest))
    return dest


def _stop_tunnel(tunnel: TunnelBackend | None) -> None:
    if tunnel is None:
        return
    stop = getattr(tunnel, "stop", None)
    if callable(stop):
        with contextlib.suppress(Exception):
            stop()


def _install_sse_wakeup_on_exit(
    server: uvicorn.Server,
    *,
    web_proc: subprocess.Popen[bytes] | None = None,
    tunnel: TunnelBackend | None = None,
) -> None:
    """Wrap uvicorn's exit handler so SSE long-polls stop on the first Ctrl+C."""
    original = server.handle_exit

    def handle_exit(sig: int, frame: FrameType | None) -> None:
        try:
            from molexp.server.shutdown import mark_shutting_down
            from molexp.services.approval_notify import close_approval_subscribers

            mark_shutting_down()
            close_approval_subscribers()
        except Exception:  # never block signal handling on a soft failure
            pass
        _stop_tunnel(tunnel)
        _stop_web_dev_server(web_proc)
        original(sig, frame)

    # Deliberate monkeypatch of a bound method — the signature above matches
    # uvicorn's exactly, so the call site is unaffected.
    server.handle_exit = handle_exit  # ty: ignore[invalid-assignment]
