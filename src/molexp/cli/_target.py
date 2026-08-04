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

from typing import TYPE_CHECKING, Annotated

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

if TYPE_CHECKING:
    from molexp.workspace import Workspace

#: Shared ``--workspace/-ws`` option (``-t/--target`` kept as a hidden,
#: back-compatible alias). Defaults to the current directory.
TargetOption = Annotated[
    str,
    typer.Option(
        "--workspace",
        "-ws",
        "--target",
        "-t",
        help="Workspace: a path, user@host:path, or @target-name (default: cwd).",
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


def open_workspace(
    target_str: str = ".",
    *,
    require_existing: bool = True,
) -> tuple[Target, Transport, FileSystem, Workspace]:
    """Resolve *target_str* and open the Workspace on the matching FileSystem.

    Local and remote targets share this path: remote roots use
    :class:`~molexp.workspace.fs_remote.RemoteFileSystem` so every Folder/Run
    I/O goes over SSH.  ``require_existing=True`` (default) fails when
    ``workspace.json`` is missing; pass ``False`` for ``init``-style create.

    Raises:
        FileNotFoundError: when *require_existing* and no workspace marker.
        typer.Exit: when the target string cannot be resolved.
    """
    from molexp.workspace import Workspace

    target, transport, fs = resolve_workspace_target(target_str)
    root = target.path
    root_str = str(root)
    if require_existing:
        marker = fs.join(root_str, "workspace.json")
        if not fs.exists(marker):
            raise FileNotFoundError(
                f"No workspace found at {root} — run molexp init {root} to create one"
            )
    ws = Workspace(root, fs=fs)
    return target, transport, fs, ws
