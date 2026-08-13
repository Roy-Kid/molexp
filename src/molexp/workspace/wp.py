"""Workspace path toolkit — ``ws.wp.mv`` / ``me.wp.mv`` for layout fixes.

Read/write path ops scoped to a :class:`~molexp.workspace.Workspace` root,
speaking through the workspace's :class:`~molexp.workspace.fs.FileSystem`
(local or remote). Designed for agent loops that fix
:func:`~molexp.workspace.validate.validate_workspace` findings (e.g. move a
``layout.stray`` directory under ``projects/.../assets/``).

Entity-aware rehoming (``Run`` → another experiment) stays on
:meth:`~molexp.workspace.folder.Folder.move_to` / curation reorg; this
module is the **raw path** surface.

Example::

    import molexp as me

    ws = me.Workspace("/path/to/ws")
    ws.wp.mv("mace-r2san-nve-verify", "projects/mace-r2san/assets/nve-verify")
    # or free functions:
    me.wp.mv(ws, "leftover", "projects/x/assets/leftover")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.workspace.fs import FileSystem
    from molexp.workspace.workspace import Workspace

__all__ = ["WorkspacePaths", "cp", "ls", "mkdir", "mv", "rm"]


class WorkspacePaths:
    """Bound path ops for one workspace (``ws.wp``)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    # ── public verbs ────────────────────────────────────────────────────

    def ls(self, path: str = ".") -> list[str]:
        """List directory entries (relative to workspace root by default)."""
        return ls(self._ws, path)

    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> str:
        """Create a directory under the workspace; return its abs path."""
        return mkdir(self._ws, path, parents=parents, exist_ok=exist_ok)

    def mv(self, src: str, dst: str) -> str:
        """Move *src* to *dst* (both workspace-relative or absolute-in-root).

        If *dst* is an existing directory, *src* is moved *into* it (basename
        preserved). Returns the absolute destination path.
        """
        return mv(self._ws, src, dst)

    def cp(self, src: str, dst: str, *, recursive: bool = True) -> str:
        """Copy *src* to *dst*. Directories require ``recursive=True``."""
        return cp(self._ws, src, dst, recursive=recursive)

    def rm(self, path: str, *, recursive: bool = False) -> None:
        """Remove a file or directory (``recursive=True`` for non-empty dirs)."""
        rm(self._ws, path, recursive=recursive)


# ── free functions (``me.wp.mv(ws, …)``) ─────────────────────────────────


def ls(workspace: Workspace, path: str = ".") -> list[str]:
    """List directory entries under *workspace*."""
    fs = _fs(workspace)
    target = _abs(workspace, path)
    if not fs.is_dir(target):
        raise NotADirectoryError(target)
    return sorted(fs.listdir(target))


def mkdir(
    workspace: Workspace,
    path: str,
    *,
    parents: bool = True,
    exist_ok: bool = True,
) -> str:
    """Create *path* under *workspace*; return absolute path."""
    fs = _fs(workspace)
    target = _abs(workspace, path)
    fs.mkdir(target, parents=parents, exist_ok=exist_ok)
    return target


def mv(workspace: Workspace, src: str, dst: str) -> str:
    """Move *src* → *dst* inside *workspace*; return absolute destination."""
    fs = _fs(workspace)
    src_abs = _abs(workspace, src)
    dst_abs = _abs(workspace, dst)
    if not fs.exists(src_abs):
        raise FileNotFoundError(src_abs)
    if fs.is_dir(dst_abs):
        dst_abs = fs.join(dst_abs, fs.basename(src_abs))
    if fs.exists(dst_abs):
        raise FileExistsError(dst_abs)
    parent = fs.dirname(dst_abs)
    if parent and not fs.exists(parent):
        fs.mkdir(parent, parents=True, exist_ok=True)
    fs.rename(src_abs, dst_abs)
    return dst_abs


def cp(workspace: Workspace, src: str, dst: str, *, recursive: bool = True) -> str:
    """Copy *src* → *dst* inside *workspace*; return absolute destination."""
    fs = _fs(workspace)
    src_abs = _abs(workspace, src)
    dst_abs = _abs(workspace, dst)
    if not fs.exists(src_abs):
        raise FileNotFoundError(src_abs)
    if fs.is_dir(dst_abs):
        dst_abs = fs.join(dst_abs, fs.basename(src_abs))
    if fs.exists(dst_abs):
        raise FileExistsError(dst_abs)
    parent = fs.dirname(dst_abs)
    if parent and not fs.exists(parent):
        fs.mkdir(parent, parents=True, exist_ok=True)
    if fs.is_dir(src_abs):
        if not recursive:
            raise IsADirectoryError(f"{src_abs} is a directory; pass recursive=True")
        fs.copytree(src_abs, dst_abs, dirs_exist_ok=False)
    else:
        fs.copy(src_abs, dst_abs)
    return dst_abs


def rm(workspace: Workspace, path: str, *, recursive: bool = False) -> None:
    """Remove *path* under *workspace*."""
    fs = _fs(workspace)
    target = _abs(workspace, path)
    if not fs.exists(target):
        raise FileNotFoundError(target)
    if fs.is_dir(target) and not recursive:
        # Non-empty dirs fail on most backends without recursive; empty ones
        # still need recursive=False support — FileSystem.remove handles it.
        children = fs.listdir(target)
        if children:
            raise OSError(f"directory not empty: {target} (pass recursive=True)")
    fs.remove(target, recursive=recursive or fs.is_dir(target))


# ── helpers ─────────────────────────────────────────────────────────────


def _fs(workspace: Workspace) -> FileSystem:
    return workspace._fs


def _abs(workspace: Workspace, path: str) -> str:
    """Resolve *path* under the workspace root; refuse escapes."""
    fs = _fs(workspace)
    root = fs.resolve(str(workspace.resolve()))
    raw = (path or ".").strip() or "."
    if raw in {".", ""}:
        return root
    if fs.is_absolute(raw):
        candidate = fs.resolve(raw)
    else:
        candidate = fs.resolve(fs.join(root, raw))
    root_n = root.rstrip("/") or "/"
    if candidate != root_n and not candidate.startswith(root_n + "/"):
        raise ValueError(f"path outside workspace root: {path!r} -> {candidate}")
    return candidate
