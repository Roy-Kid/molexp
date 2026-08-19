"""Single filesystem tree walk — local and remote share one implementation.

Routes and UI discovery must not re-implement ``Path.iterdir`` vs
``workspace.fs.listdir``. Every recursive listing goes through
:func:`list_tree_children`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from molexp.workspace.fs import FileSystem


@dataclass
class FsTreeNode:
    """One node in a filesystem tree rooted at a walk root."""

    name: str
    rel_path: str
    abs_path: str
    type: str  # "file" | "folder"
    size: int | None = None
    modified: float | None = None
    children: list[FsTreeNode] = field(default_factory=list)


IgnoreFn = Callable[[str, bool], bool]
"""``ignore(rel_path, is_dir) -> True`` to skip a child."""


def list_tree_children(
    fs: FileSystem,
    root: str,
    *,
    max_depth: int = 8,
    ignore: IgnoreFn | None = None,
) -> list[FsTreeNode]:
    """Return the children of *root*, each expanded up to *max_depth*.

    *root* itself is not included — callers get a list of top-level
    entries (same shape as ``GET /api/workspace/files`` ``children``).

    Depth is measured from *root*: ``max_depth=0`` yields empty children
    lists for every top-level directory; ``max_depth=1`` lists one level
    under each top-level folder.
    """
    root_norm = str(root).rstrip("/") or "/"

    def _rel(abs_path: str) -> str:
        node = abs_path.rstrip("/") or "/"
        if node == root_norm:
            return ""
        prefix = root_norm + "/"
        if not node.startswith(prefix):
            # basename fallback when fs.join produces an unexpected shape
            return fs.basename(abs_path)
        return node[len(prefix) :]

    def build(node_path: str, depth: int, *, visited: set[str]) -> FsTreeNode:
        try:
            real = fs.resolve(node_path)
        except OSError:
            real = node_path
        name = fs.basename(node_path) or node_path
        rel = _rel(node_path)

        if real in visited:
            return FsTreeNode(
                name=name,
                rel_path=rel,
                abs_path=node_path,
                type="folder",
                children=[],
            )
        visited.add(real)

        try:
            st = fs.stat(node_path)
            is_file = st.is_file
            size = st.size if is_file else None
            mtime = st.mtime
        except OSError:
            return FsTreeNode(
                name=name,
                rel_path=rel,
                abs_path=node_path,
                type="folder",
                children=[],
            )

        node = FsTreeNode(
            name=name,
            rel_path=rel,
            abs_path=node_path,
            type="file" if is_file else "folder",
            size=size,
            modified=mtime,
            children=[],
        )
        if is_file or depth >= max_depth:
            return node

        try:
            names = fs.listdir(node_path)
        except OSError:
            names = []

        children: list[FsTreeNode] = []
        for child_name in sorted(names):
            child_path = fs.join(node_path, child_name)
            child_rel = _rel(child_path)
            # Avoid an extra is_dir RTT for ignore: dotted basenames that are
            # not hidden dirs are treated as files (same heuristic as workspace route).
            looks_like_file = "." in child_name and not child_name.startswith(".")
            if ignore is not None and ignore(child_rel, not looks_like_file):
                continue
            children.append(build(child_path, depth + 1, visited=visited))
        children.sort(key=lambda c: (c.type == "file", c.name))
        node.children = children
        return node

    if not fs.exists(root_norm):
        return []

    try:
        top_names = fs.listdir(root_norm)
    except OSError:
        return []

    out: list[FsTreeNode] = []
    visited: set[str] = set()
    for name in sorted(top_names):
        child = fs.join(root_norm, name)
        rel = _rel(child)
        looks_like_file = "." in name and not name.startswith(".")
        if ignore is not None and ignore(rel, not looks_like_file):
            continue
        out.append(build(child, 0, visited=visited))
    out.sort(key=lambda c: (c.type == "file", c.name))
    return out


def tree_to_run_file_dicts(nodes: list[FsTreeNode]) -> list[dict[str, Any]]:
    """Map :class:`FsTreeNode` trees to the run-files API node shape."""

    def one(n: FsTreeNode) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": n.name,
            "relPath": n.rel_path,
            "type": n.type,
            "size": n.size,
            "modified": n.modified,
        }
        if n.type == "folder":
            d["children"] = [one(c) for c in n.children]
        return d

    return [one(n) for n in nodes]


def tree_to_workspace_file_dicts(nodes: list[FsTreeNode]) -> list[dict[str, Any]]:
    """Map :class:`FsTreeNode` trees to the workspace-files API child shape."""

    def one(n: FsTreeNode) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": n.abs_path,
            "name": n.name,
            "path": n.abs_path,
            "type": n.type,
            "size": n.size,
            "modified": n.modified,
            "children": [one(c) for c in n.children] if n.type == "folder" else [],
        }
        return d

    return [one(n) for n in nodes]


__all__ = [
    "FsTreeNode",
    "IgnoreFn",
    "list_tree_children",
    "tree_to_run_file_dicts",
    "tree_to_workspace_file_dicts",
]
