"""``molexp workspace`` — unified local/remote workspace entry point.

Usage::

    molexp workspace [TARGET] <command>

TARGET can be:
- (omitted) → current directory
- ``/local/path`` → local workspace
- ``user@host:/path`` → remote workspace (SCP-style)
- ``@target-name`` → registered compute target
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from molexp.workspace.fs import FileSystem
from molexp.workspace.target import (
    LocalTarget,
    RemoteTarget,
    Target,
    TargetNotFound,
    parse_target,
    resolve_target,
    target_to_filesystem,
    target_to_transport,
)

workspace_app = typer.Typer(
    name="workspace",
    help="Manage a molexp workspace (local or remote).",
    invoke_without_command=True,
)


@workspace_app.callback()
def _workspace_callback(
    ctx: typer.Context,
    target: Annotated[
        str | None,
        typer.Argument(help="Workspace target: path, user@host:path, or @target-name"),
    ] = None,
    interactive: Annotated[  # noqa: F811
        bool,
        typer.Option("-i", "--interactive", help="Enter interactive session mode"),
    ] = False,
) -> None:
    """Resolve the workspace target and store in context for subcommands."""
    target_raw = target or "."
    ctx.ensure_object(dict)

    try:
        resolved, transport = resolve_target(target_raw)
    except TargetNotFound as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    fs = target_to_filesystem(resolved)

    ctx.obj["target_raw"] = target_raw
    ctx.obj["target"] = resolved
    ctx.obj["transport"] = transport
    ctx.obj["fs"] = fs
    ctx.obj["interactive"] = interactive

    # -i without subcommand → enter interactive REPL directly
    if ctx.invoked_subcommand is None:
        if interactive:
            from molexp.cli.workspace.interactive import run_interactive

            run_interactive(resolved, transport, target_raw)
            raise typer.Exit()
        # No -i and no subcommand → show help
        rprint("[yellow]Missing command. Use -i for interactive mode, or see --help.[/yellow]")
        raise typer.Exit(1)


def _get_ctx_target(ctx: typer.Context) -> Target:
    return ctx.obj["target"]


def _get_ctx_transport(ctx: typer.Context):
    return ctx.obj["transport"]


def _load_workspace(target: Target, *, fs: FileSystem | None = None) -> Workspace:  # noqa: F821
    """Load a molexp Workspace from a target.

    When *fs* is provided, the workspace layer operates through it
    (local or remote transparently).
    """
    from molexp.workspace import Workspace
    from molexp.workspace.fs import FileSystem as FS

    if fs is not None:
        return Workspace(str(target), fs=fs)
    if isinstance(target, LocalTarget):
        return Workspace(target.path)
    # Remote without fs — try constructing with path
    return Workspace(str(target), fs=FS())  # fallback: error at first I/O


class RemoteWorkspaceError(Exception):
    """Full workspace CRUD on remote targets is not yet implemented.
    Use ``exec``, ``shell``, ``sync`` for remote operations.
    """

    def __init__(self, target: RemoteTarget) -> None:
        super().__init__(
            f"Remote workspace CRUD not yet supported for {target}. "
            "Use 'exec', 'shell', or 'sync' subcommands."
        )


from molexp.cli._common import rprint  # noqa: E402

# Import subcommand modules for side-effect registration.
from molexp.cli.workspace import (  # noqa: E402
    explore,
    interactive,
    lifecycle,
    monitor,
    resources,
    run,
    serve,
    sync,
)
