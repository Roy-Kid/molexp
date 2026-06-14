"""Deprecated shim — the tree explorer moved to :mod:`molexp.cli.tui`.

Import :class:`TreeMonitor` and friends from ``molexp.cli.tui`` (public
surface) instead. This module re-exports the old names so external
imports keep working, and emits a :class:`DeprecationWarning` on import.
"""

from __future__ import annotations

import warnings

from molexp.cli.tui import (
    SHORT_ID_LEN,
    NodeKind,
    NodePath,
    TreeMonitor,
    TreeNode,
    TreeNodeRef,
    VisibleRow,
    build_tree,
    flatten,
    node_path_str,
)
from molexp.cli.tui.rendering import _DeleteDialog  # noqa: F401
from molexp.cli.tui.tree_model import (  # noqa: F401
    _as_dict,
    _as_str,
    _as_str_dict,
    _short_exec_label,
    _short_id,
)
from molexp.cli.tui.tree_monitor import (  # noqa: F401
    _collect_targets,
    _describe_target,
    _execute_delete,
    _prepare_dialog,
    _UIState,
)

warnings.warn(
    "molexp.tree_monitor is deprecated; import from molexp.cli.tui instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SHORT_ID_LEN",
    "NodeKind",
    "NodePath",
    "TreeMonitor",
    "TreeNode",
    "TreeNodeRef",
    "VisibleRow",
    "build_tree",
    "flatten",
    "node_path_str",
]
