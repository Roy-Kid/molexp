"""``molexp git {checkpoint,rebuild,push}`` — workspace→git projection commands.

Thin CLI surface over the **shared** backend in
:mod:`molexp.workspace.git_projection`: these commands import and call the exact
same ``checkpoint`` / ``rebuild`` / ``push`` symbols the server's ``/api/git/*``
routes call (Python ≡ UI — one backend code path, never duplicated logic).

``checkpoint`` / ``rebuild`` materialize the bare object DB locally (cheap,
ungated); ``push`` is outward-facing and is the gated curation capability.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from molexp.cli._common import get_workspace, rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.git_projection import (
    checkpoint,
    default_object_db_path,
    push,
    rebuild,
)
from molexp.workspace.target import LocalTarget

git_app = typer.Typer(help="Git checkpoint projection of the workspace", no_args_is_help=True)


def _remote_only(verb: str) -> NoReturn:
    rprint(f"[red]Error:[/red] `molexp git {verb}` only supports a local workspace.")
    raise typer.Exit(1)


def _local_workspace(target_spec: str):
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("git")
    return get_workspace(target.path if target.path != Path.cwd() else None)


@git_app.command("checkpoint")
def git_checkpoint(target_spec: TargetOption = ".") -> None:
    """Project the workspace onto git objects + refs/molexp/* (local, ungated)."""
    ws = _local_workspace(target_spec)
    result = asyncio.run(checkpoint(ws))
    rprint(
        f"[green]OK[/green] checkpointed {len(result.runs)} run(s) → {default_object_db_path(ws)}"
    )


@git_app.command("rebuild")
def git_rebuild(target_spec: TargetOption = ".") -> None:
    """Erase + deterministically re-derive the projection from authoritative files."""
    ws = _local_workspace(target_spec)
    result = asyncio.run(rebuild(ws))
    rprint(f"[green]OK[/green] rebuilt projection for {len(result.runs)} run(s)")


@git_app.command("push")
def git_push(
    remote: Annotated[str, typer.Argument(help="Git remote URL or path")],
    target_spec: TargetOption = ".",
) -> None:
    """Push refs/molexp/* to a git remote (outward-facing)."""
    ws = _local_workspace(target_spec)
    asyncio.run(push(ws, remote=remote))
    rprint(f"[green]OK[/green] pushed refs/molexp/* → {remote}")
