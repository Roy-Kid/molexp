"""``molexp target`` — manage compute targets on a workspace.

A *compute target* is a named (Transport × Scheduler) pair stored in the
workspace's ``workspace.json``.  ``--host`` (transport axis) and
``--scheduler`` (dispatch axis) are independent — leave both off for the
local-shell default, set ``--host`` to push commands over SSH, set
``--scheduler`` to dispatch through SLURM/PBS/LSF instead of plain shell.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Literal

import typer

from molexp.workspace import (
    ComputeTarget,
    add_target,
    get_target,
    list_targets,
    remove_target,
    to_transport,
)

from . import app
from ._common import get_workspace, rprint

target_app = typer.Typer(
    name="target",
    help="Manage compute targets (where and how runs execute)",
    no_args_is_help=True,
)
app.add_typer(target_app, name="target")


@target_app.command("list")
def list_cmd(
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """List all compute targets registered on the workspace."""
    ws = get_workspace(path)
    targets = list_targets(ws)
    if not targets:
        rprint("[yellow]No compute targets registered.[/yellow]")
        rprint(
            "  Add one with: "
            "[cyan]molexp target add NAME --scratch /path "
            "[--host user@host] [--scheduler slurm|pbs|lsf|shell][/cyan]"
        )
        return

    for t in targets:
        location = f"[cyan]{t.host}[/cyan]" if t.host else "[dim]local[/dim]"
        rprint(
            f"[bold]{t.name}[/bold]  "
            f"location={location}  "
            f"scheduler=[magenta]{t.scheduler}[/magenta]  "
            f"scratch={t.scratch_root}"
        )


@target_app.command("add")
def add_cmd(
    name: Annotated[str, typer.Argument(help="Target name (used by --target)")],
    scratch: Annotated[
        str,
        typer.Option("--scratch", help="Absolute scratch root on the target's filesystem"),
    ],
    scheduler: Annotated[
        str,
        typer.Option(
            "--scheduler",
            help="Dispatch axis: local | slurm | pbs | lsf",
        ),
    ] = "local",
    host: Annotated[
        str | None,
        typer.Option("--host", help="user@host for SSH transport (omit for local)"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="SSH port (default: ssh_config)"),
    ] = None,
    identity: Annotated[
        str | None,
        typer.Option("--identity", help="SSH identity file (default: ssh-agent / ssh_config)"),
    ] = None,
    ssh_opt: Annotated[
        list[str] | None,
        typer.Option(
            "--ssh-opt",
            help="Extra ssh argv (repeatable, e.g. --ssh-opt -o --ssh-opt ServerAliveInterval=30)",
        ),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Add a compute target to the workspace."""
    # Narrow the free-form ``str`` to the Literal arm expected by
    # ``ComputeTarget.scheduler`` so ty can prove the call site.
    if scheduler == "local":
        narrowed_scheduler: Literal["local", "slurm", "pbs", "lsf"] = "local"
    elif scheduler == "slurm":
        narrowed_scheduler = "slurm"
    elif scheduler == "pbs":
        narrowed_scheduler = "pbs"
    elif scheduler == "lsf":
        narrowed_scheduler = "lsf"
    else:
        rprint(
            f"[red]Invalid scheduler {scheduler!r}[/red] — must be one of: local, slurm, pbs, lsf"
        )
        raise typer.Exit(2)

    ws = get_workspace(path)
    try:
        target = ComputeTarget(
            name=name,
            host=host,
            port=port,
            identity_file=identity,
            ssh_opts=list(ssh_opt or []),
            scheduler=narrowed_scheduler,
            scratch_root=scratch,
        )
    except ValueError as exc:
        rprint(f"[red]Invalid target:[/red] {exc}")
        raise typer.Exit(2) from exc

    try:
        add_target(ws, target)
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    location = host or "local"
    rprint(
        f"[green]OK[/green] Added target [bold]{name}[/bold] "
        f"({location}, scheduler={scheduler}, scratch={scratch})"
    )


@target_app.command("remove")
def remove_cmd(
    name: Annotated[str, typer.Argument(help="Target name to remove")],
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Remove a compute target from the workspace."""
    ws = get_workspace(path)
    try:
        remove_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] Removed target [bold]{name}[/bold]")


@target_app.command("test")
def test_cmd(
    name: Annotated[str, typer.Argument(help="Target name to verify")],
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Verify connectivity to a target — runs a tiny round-trip via the transport.

    For local targets this is essentially a no-op (mkdir + write + read).
    For remote targets it runs ``ssh host true``, ``mkdir -p`` of the scratch
    root, and a 1-byte file round-trip via rsync.  Catches the most common
    misconfigurations (wrong host, missing key, no rsync) before any real job
    is submitted.
    """
    ws = get_workspace(path)
    try:
        target = get_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if target.is_remote:
        if shutil.which("ssh") is None:
            rprint("[red]ssh binary not found in PATH — install OpenSSH client[/red]")
            raise typer.Exit(2)
        if shutil.which("rsync") is None:
            rprint("[red]rsync binary not found in PATH — install rsync[/red]")
            raise typer.Exit(2)

    transport = to_transport(target)
    rprint(f"[bold]Testing target {name!r}...[/bold]")

    # 1. Run a trivial command.
    try:
        result = transport.run(["true"], timeout=15)
        if result.returncode != 0:
            rprint(f"[red]Command 'true' returned {result.returncode}: {result.stderr}[/red]")
            raise typer.Exit(1)
    except Exception as exc:
        rprint(f"[red]Transport.run() failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint("  [green]ok[/green] command execution")

    # 2. mkdir scratch root.
    try:
        transport.mkdir(target.scratch_root, parents=True, exist_ok=True)
    except Exception as exc:
        rprint(f"[red]mkdir {target.scratch_root!r} failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint(f"  [green]ok[/green] mkdir {target.scratch_root}")

    # 3. Write + read 1-byte file (round-trip).
    probe = f"{target.scratch_root.rstrip('/')}/.molexp-target-test"
    try:
        transport.write_text(probe, "x")
        if transport.read_text(probe) != "x":
            rprint("[red]Round-trip mismatch on probe file[/red]")
            raise typer.Exit(1)
        transport.remove(probe)
    except Exception as exc:
        rprint(f"[red]Probe round-trip failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint("  [green]ok[/green] file round-trip")

    rprint(f"[green]Target {name!r} is reachable.[/green]")
