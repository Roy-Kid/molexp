"""Terminal UI components for the molexp CLI.

Hosts the ``molexp explore`` tree explorer, split into:

- :mod:`~molexp.cli.tui.tree_model` — workspace tree data model
  (:class:`TreeNode`, :func:`build_tree`, :func:`flatten`).
- :mod:`~molexp.cli.tui.rendering` — pure Rich rendering.
- :mod:`~molexp.cli.tui.tree_monitor` — UI state, input handling, and
  the :class:`TreeMonitor` live loop.

plus :mod:`~molexp.cli.tui.run_monitor` — the :class:`RunMonitor`
lifecycle controller for the full-screen molq run dashboard.

The legacy import path ``molexp.tree_monitor`` is kept as a thin
deprecation shim over this package.
"""

from .run_monitor import RunMonitor
from .tree_model import (
    SHORT_ID_LEN,
    NodeKind,
    NodePath,
    TreeNode,
    TreeNodeRef,
    VisibleRow,
    build_tree,
    flatten,
    node_path_str,
)
from .tree_monitor import TreeMonitor

__all__ = [
    "SHORT_ID_LEN",
    "NodeKind",
    "NodePath",
    "RunMonitor",
    "TreeMonitor",
    "TreeNode",
    "TreeNodeRef",
    "VisibleRow",
    "build_tree",
    "flatten",
    "node_path_str",
]
