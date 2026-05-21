"""``molexp target`` — list/add/remove/test compute-target registry.

Compute targets live on ``WorkspaceMetadata.targets`` as immutable
:class:`molexp.workspace.models.ComputeTarget` records.  This CLI is a
thin user-facing wrapper around the CRUD helpers in
:mod:`molexp.workspace.targets`.

Each subcommand is fully workspace-rooted via the ``--path`` option,
because the registry is per-workspace state.  No global registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from molexp.cli._common import rprint

target_app = typer.Typer(
    name="target",
    help="Manage compute targets (list / add / remove / test).",
    no_args_is_help=True,
)


_VALID_SCHEDULERS = ("local", "slurm", "pbs", "lsf")


def _open_workspace(path: Path):
    """Load the workspace at *path*, exiting 1 with a helpful message on failure."""
    from molexp.workspace import Workspace

    try:
        return Workspace.load(path)
    except Exception as exc:
        rprint(f"[red]Failed to open workspace at {path}:[/red] {exc}")
        raise typer.Exit(1) from exc


@target_app.command("list")
def list_cmd(
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Workspace path"),
    ],
) -> None:
    """List all registered compute targets."""
    from molexp.workspace.targets import list_targets

    ws = _open_workspace(path)
    targets = list_targets(ws)
    if not targets:
        rprint("[dim]No compute targets registered.[/dim]")
        return
    for t in targets:
        host = t.host or "(local)"
        rprint(
            f"  [bold]{t.name}[/bold]  scheduler={t.scheduler}  "
            f"host={host}  scratch={t.scratch_root}"
        )


@target_app.command("add")
def add_cmd(
    name: Annotated[str, typer.Argument(help="Target name (must be unique)")],
    scratch: Annotated[
        str, typer.Option("--scratch", help="Scratch root on the target's filesystem")
    ],
    scheduler: Annotated[
        str,
        typer.Option(
            "--scheduler",
            help=f"One of: {', '.join(_VALID_SCHEDULERS)}",
        ),
    ] = "local",
    host: Annotated[
        str | None,
        typer.Option("--host", help="SSH host (omit for local target)"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="SSH port"),
    ] = None,
    identity_file: Annotated[
        str | None,
        typer.Option("--identity-file", help="SSH identity file"),
    ] = None,
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = Path.cwd(),
) -> None:
    """Register a new compute target on the workspace."""
    if scheduler not in _VALID_SCHEDULERS:
        rprint(
            f"[red]Invalid scheduler {scheduler!r}.[/red] "
            f"Must be one of: {', '.join(_VALID_SCHEDULERS)}"
        )
        # exit code 2 matches typer's "invalid usage" convention; tests rely on it
        raise typer.Exit(2)

    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.targets import add_target

    ws = _open_workspace(path)
    target = ComputeTarget(
        name=name,
        host=host,
        port=port,
        identity_file=identity_file,
        scheduler=scheduler,  # type: ignore[arg-type]
        scratch_root=scratch,
    )
    try:
        add_target(ws, target)
    except ValueError as exc:
        # already exists → exit 1 per tests
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    rprint(
        f"[green]OK[/green] Added target {name}  "
        f"scheduler={target.scheduler}  "
        f"host={target.host or '(local)'}  "
        f"scratch={target.scratch_root}"
    )


@target_app.command("remove")
def remove_cmd(
    name: Annotated[str, typer.Argument(help="Target name to remove")],
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = Path.cwd(),
) -> None:
    """Remove a compute target from the workspace registry."""
    from molexp.workspace.targets import remove_target

    ws = _open_workspace(path)
    try:
        remove_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]No target named {name!r}[/red]")
        raise typer.Exit(1) from exc

    rprint(f"[green]OK[/green] Removed target {name}")


@target_app.command("test")
def test_cmd(
    name: Annotated[str, typer.Argument(help="Target name to probe")],
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = Path.cwd(),
) -> None:
    """Probe a target: command exec → scratch mkdir → file round-trip.

    Runs three minimal smoke checks against the target's transport, and
    reports each one as ``ok <step>`` / ``fail <step>: <reason>``.
    """
    import uuid

    from molexp.workspace.targets import get_target, to_transport

    ws = _open_workspace(path)
    try:
        target = get_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]No target named {name!r}[/red]")
        raise typer.Exit(1) from exc

    transport = to_transport(target)

    # Step 1: command execution
    try:
        result = transport.run(["echo", "molexp-target-test"])
        if result.returncode != 0:
            rprint(f"[red]fail command execution:[/red] returncode={result.returncode}")
            raise typer.Exit(1)
        rprint("[green]ok command execution[/green]")
    except Exception as exc:
        rprint(f"[red]fail command execution:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Step 2: scratch mkdir
    try:
        transport.mkdir(target.scratch_root, parents=True, exist_ok=True)
        rprint(f"[green]ok mkdir[/green] {target.scratch_root}")
    except Exception as exc:
        rprint(f"[red]fail mkdir:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Step 3: write → read round-trip under scratch
    probe_name = f".molexp-target-probe-{uuid.uuid4().hex[:8]}"
    probe_path = target.scratch_root.rstrip("/") + "/" + probe_name
    payload = "molexp target probe\n"
    try:
        transport.write_text(probe_path, payload)
        got = transport.read_text(probe_path)
        transport.remove(probe_path)
        if got != payload:
            rprint("[red]fail file round-trip:[/red] payload mismatch")
            raise typer.Exit(1)
        rprint("[green]ok file round-trip[/green]")
    except Exception as exc:
        rprint(f"[red]fail file round-trip:[/red] {exc}")
        raise typer.Exit(1) from exc
