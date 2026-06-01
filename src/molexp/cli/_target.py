"""Shared ``-t/--target`` option + resolver for top-level CLI commands.

Replaces the old ``workspace`` group callback's ``ctx.obj`` plumbing: a
top-level command declares the :data:`TargetOption` parameter and calls
:func:`resolve_workspace_target` to obtain the same ``(target, transport, fs)``
triple the callback used to stash on the context. Local is the zero-config
default (``-t .``); remote is ``-t user@host:/path`` or a registered
``-t @target-name`` (resolved against the cwd workspace's compute-target
registry — an improvement over the old callback, which passed no workspace and
so could not resolve ``@name`` at all).
"""

from __future__ import annotations

from typing import Annotated

import typer
from molq.transport import Transport

from molexp.workspace.fs import FileSystem
from molexp.workspace.target import (
    Target,
    TargetNeedsResolution,
    TargetNotFound,
    resolve_target,
    target_to_filesystem,
)

#: Shared ``-t/--target`` option. Defaults to the current directory.
TargetOption = Annotated[
    str,
    typer.Option(
        "-t",
        "--target",
        help="Workspace target: path, user@host:path, or @target-name (default: cwd).",
    ),
]


def resolve_workspace_target(target_str: str = ".") -> tuple[Target, Transport, FileSystem]:
    """Resolve a target string into ``(target, transport, fs)``.

    Wraps :func:`molexp.workspace.target.resolve_target` +
    :func:`target_to_filesystem` so every top-level command shares one
    local/remote resolution path. ``@name`` targets are looked up in the cwd
    workspace's compute-target registry.

    Args:
        target_str: Target spec; ``"."`` (default) -> local cwd workspace,
            ``user@host:/path`` -> remote, ``@name`` -> registered target.

    Returns:
        ``(resolved_target, transport, filesystem)``.

    Raises:
        typer.Exit: code 1 if the target cannot be resolved, after printing
            the error.
    """
    spec = target_str or "."

    ws = None
    if spec.startswith("@"):
        # ``@name`` needs a workspace to look up the compute-target registry.
        from molexp.workspace import Workspace

        try:
            ws = Workspace.load(".")
        except Exception:
            ws = None

    try:
        resolved, transport = resolve_target(spec, ws)
    except (TargetNotFound, TargetNeedsResolution) as exc:
        from molexp.cli._common import rprint

        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    fs = target_to_filesystem(resolved)
    return resolved, transport, fs
