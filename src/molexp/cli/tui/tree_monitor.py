"""Interactive tree monitor for ``molexp explore``.

Reads a workspace, builds a collapsible tree
(workspace → project → experiment → run → execution), and renders a
full-screen terminal UI with breadcrumb navigation, multi-select, and
a cancel-then-delete confirmation flow.

Keybindings (list mode)::

    ↑ / ↓          navigate
    Enter          expand / collapse selected node
    →              open the right-side detail panel (close it if already open)
    ←              close the detail panel if open, else collapse the selected node
    space / x      toggle multi-select on selected node
    a              select all currently-visible nodes
    A              clear multi-selection
    d              open delete confirmation for selected / multi-selected nodes
    q / Ctrl-C     quit

Keybindings (confirm dialog)::

    y              confirm delete
    n / Esc        cancel

Only terminal runs/executions are deleted silently.  Running ones get
their cancel path inspected first (see :mod:`molexp.plugins.submit_molq.cancel`):
cancellable ones are cancelled then deleted; uncancellable ones stay
with a warning printed after the dialog closes.

The tree data model lives in :mod:`molexp.cli.tui.tree_model`; pure
Rich rendering in :mod:`molexp.cli.tui.rendering`. This module owns UI
state, terminal input, the delete flow, and the live render loop.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live

from molexp.plugins.submit_molq.cancel import try_cancel
from molexp.workspace import Experiment, Project, Run, Workspace

from .rendering import (
    _DeleteDialog,
    _render_breadcrumb,
    _render_detail,
    _render_dialog,
    _render_tree,
)
from .tree_model import NodePath, TreeNode, VisibleRow, build_tree, flatten

# ── UI state ──────────────────────────────────────────────────────────────────


class _UIState:
    """Thread-safe UI state: cursor, expansion, multi-select, dialog."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._selected = 0
        self._visible_count = 0
        self._expanded: set[NodePath] = set()
        self._multi: set[NodePath] = set()
        self._dialog: _DeleteDialog | None = None
        self._detail_open = False
        self._quit = False
        self._stale_warnings: list[str] = []

    # ── Cursor ──
    @property
    def selected(self) -> int:
        with self._lock:
            return self._selected

    def move(self, delta: int) -> None:
        with self._lock:
            if self._visible_count == 0:
                return
            self._selected = max(0, min(self._visible_count - 1, self._selected + delta))

    def set_visible_count(self, n: int) -> None:
        with self._lock:
            self._visible_count = n
            if self._selected >= n:
                self._selected = max(0, n - 1)

    # ── Expansion ──
    def toggle_expand(self, node_id: NodePath) -> None:
        with self._lock:
            if node_id in self._expanded:
                self._expanded.discard(node_id)
            else:
                self._expanded.add(node_id)

    def expand(self, node_id: NodePath) -> None:
        with self._lock:
            self._expanded.add(node_id)

    def collapse(self, node_id: NodePath) -> None:
        with self._lock:
            self._expanded.discard(node_id)

    def is_expanded(self, node_id: NodePath) -> bool:
        with self._lock:
            return node_id in self._expanded

    def expanded_snapshot(self) -> set[NodePath]:
        with self._lock:
            return set(self._expanded)

    # ── Multi-select ──
    def toggle_multi(self, node_id: NodePath) -> None:
        with self._lock:
            if node_id in self._multi:
                self._multi.discard(node_id)
            else:
                self._multi.add(node_id)

    def select_all_visible(self, visible: list[VisibleRow]) -> None:
        with self._lock:
            for r in visible:
                self._multi.add(r.node.node_id)

    def clear_multi(self) -> None:
        with self._lock:
            self._multi.clear()

    def multi_snapshot(self) -> set[NodePath]:
        with self._lock:
            return set(self._multi)

    def drop_missing(self, present: set[NodePath]) -> None:
        """Purge multi-select/expansion entries for nodes no longer in the tree."""
        with self._lock:
            self._multi.intersection_update(present)
            self._expanded.intersection_update(present)

    # ── Detail panel ──
    def detail_open(self) -> bool:
        with self._lock:
            return self._detail_open

    def toggle_detail(self) -> None:
        with self._lock:
            self._detail_open = not self._detail_open

    def close_detail(self) -> None:
        with self._lock:
            self._detail_open = False

    # ── Dialog / quit / warnings ──
    def open_dialog(self, dialog: _DeleteDialog) -> None:
        with self._lock:
            self._dialog = dialog

    def close_dialog(self) -> None:
        with self._lock:
            self._dialog = None

    def current_dialog(self) -> _DeleteDialog | None:
        with self._lock:
            return self._dialog

    def quit(self) -> None:
        with self._lock:
            self._quit = True

    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def add_warnings(self, messages: list[str]) -> None:
        with self._lock:
            self._stale_warnings.extend(messages)

    def drain_warnings(self) -> list[str]:
        with self._lock:
            out = list(self._stale_warnings)
            self._stale_warnings.clear()
            return out


# ── Delete flow ───────────────────────────────────────────────────────────────


def _collect_targets(root: TreeNode, ui: _UIState, cursor_node: TreeNode | None) -> list[TreeNode]:
    multi = ui.multi_snapshot()
    if multi:
        # Preserve tree order
        ordered: list[TreeNode] = []

        def walk(n: TreeNode) -> None:
            if n.node_id in multi:
                ordered.append(n)
            for c in n.children:
                walk(c)

        walk(root)
        return ordered
    if cursor_node is not None and cursor_node.kind != "workspace":
        return [cursor_node]
    return []


def _describe_target(node: TreeNode) -> str:
    kind = node.kind
    if kind == "run":
        return f"run {node.display_label}"
    if kind == "execution":
        return node.display_label
    return f"{kind} {node.display_label}"


def _prepare_dialog(targets: list[TreeNode]) -> _DeleteDialog:
    """Classify each target's cancel prospects for display."""
    lines: list[tuple[str, str, str]] = []
    for node in targets:
        running = (node.status or "").lower() in ("running", "pending")
        if node.kind == "run" and running:
            assert isinstance(node.ref, Run)
            from molexp.plugins.submit_molq.cancel import classify

            plan = classify(node.ref)
            if plan.kind == "none":
                lines.append(("!", _describe_target(node), f"uncancellable: {plan.detail}"))
            else:
                lines.append(("⟳", _describe_target(node), f"cancel via {plan.kind}"))
        elif node.kind == "execution" and running:
            # Executions share cancel path with their parent run.
            assert isinstance(node.ref, tuple) and isinstance(node.ref[0], Run)
            run = node.ref[0]
            from molexp.plugins.submit_molq.cancel import classify

            plan = classify(run)
            if plan.kind == "none":
                lines.append(("!", _describe_target(node), f"uncancellable: {plan.detail}"))
            else:
                lines.append(("⟳", _describe_target(node), f"cancel via {plan.kind}"))
        else:
            lines.append(("✓", _describe_target(node), ""))
    return _DeleteDialog(targets=targets, plan_lines=lines)


def _execute_delete(dialog: _DeleteDialog) -> list[str]:
    """Run the dialog's plan.  Returns post-action warning messages."""
    warnings: list[str] = []
    for node in dialog.targets:
        status_lower = (node.status or "").lower()
        running = status_lower in ("running", "pending")
        try:
            if node.kind == "run":
                assert isinstance(node.ref, Run)
                run = node.ref
                if running:
                    msg = try_cancel(run)
                    if msg is not None:
                        warnings.append(msg)
                        continue  # do NOT delete if cancel failed
                run.experiment.remove_run(run.id)
            elif node.kind == "execution":
                assert (
                    isinstance(node.ref, tuple)
                    and isinstance(node.ref[0], Run)
                    and isinstance(node.ref[1], str)
                )
                run = node.ref[0]
                exec_id = node.ref[1]
                if running:
                    msg = try_cancel(run)
                    if msg is not None:
                        warnings.append(msg)
                        continue
                run.delete_execution(exec_id)
            elif node.kind == "experiment":
                assert isinstance(node.ref, Experiment)
                exp = node.ref
                exp.project.remove_experiment(exp.id)
            elif node.kind == "project":
                assert isinstance(node.ref, Project)
                proj = node.ref
                proj.workspace.remove_project(proj.id)
        except Exception as exc:  # pragma: no cover - surface anything unexpected
            warnings.append(f"delete failed for {_describe_target(node)}: {exc}")
    return warnings


# ── Monitor ───────────────────────────────────────────────────────────────────


class TreeMonitor:
    """Full-screen tree monitor for a workspace."""

    def __init__(
        self,
        *,
        project_filter: str | None = None,
        experiment_filter: str | None = None,
        refresh_interval: float = 2.0,
        console: Console | None = None,
    ) -> None:
        self._project_filter = project_filter
        self._experiment_filter = experiment_filter
        self._refresh_interval = refresh_interval
        self._console = console or Console()

    def watch(self, workspace: Workspace) -> list[str]:
        """Open the monitor.  Returns any post-close warnings for the CLI."""
        ui = _UIState()
        stop = threading.Event()

        def build() -> tuple[TreeNode, list[VisibleRow]]:
            tree = build_tree(
                workspace,
                project_filter=self._project_filter,
                experiment_filter=self._experiment_filter,
            )
            ids_present: set[NodePath] = set()

            def collect(n: TreeNode) -> None:
                ids_present.add(n.node_id)
                for c in n.children:
                    collect(c)

            collect(tree)
            ui.drop_missing(ids_present)
            rows = flatten(tree, ui.expanded_snapshot())
            ui.set_visible_count(len(rows))
            return tree, rows

        def render(tree: TreeNode, rows: list[VisibleRow]) -> Layout:
            dialog = ui.current_dialog()
            crumb = _render_breadcrumb(tree, rows, ui.selected)
            tree_panel = _render_tree(rows, ui.selected, ui.multi_snapshot())

            if ui.detail_open():
                current = rows[ui.selected].node if rows else None
                detail_panel = _render_detail(current)
                body: Layout = Layout(name="body")
                body.split_row(
                    Layout(tree_panel, name="tree", ratio=3),
                    Layout(detail_panel, name="detail", ratio=2),
                )
            else:
                body = Layout(tree_panel, name="body", ratio=1)

            layout = Layout()
            if dialog is not None:
                dlg_panel = _render_dialog(dialog)
                dlg_height = max(7, len(dialog.plan_lines) + 7)
                layout.split_column(
                    Layout(crumb, name="crumb", size=3),
                    body,
                    Layout(dlg_panel, name="dialog", size=dlg_height),
                )
            else:
                layout.split_column(
                    Layout(crumb, name="crumb", size=3),
                    body,
                )
            return layout

        reader = threading.Thread(
            target=self._read_keys,
            args=(ui, stop, build),
            daemon=True,
        )
        reader.start()

        try:
            tree, rows = build()
            with Live(
                render(tree, rows),
                console=self._console,
                refresh_per_second=20,
                screen=True,
            ) as live:
                last = time.monotonic()
                while not ui.should_quit():
                    if time.monotonic() - last >= self._refresh_interval:
                        tree, rows = build()
                        last = time.monotonic()
                    else:
                        rows = flatten(tree, ui.expanded_snapshot())
                        ui.set_visible_count(len(rows))
                    live.update(render(tree, rows))
                    time.sleep(0.05)
        finally:
            stop.set()
            reader.join(timeout=2.0)

        return ui.drain_warnings()

    def _read_keys(
        self,
        ui: _UIState,
        stop: threading.Event,
        build: Callable[[], tuple[TreeNode, list[VisibleRow]]],
    ) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                ch = os.read(fd, 1).decode("utf-8", errors="replace")

                dialog = ui.current_dialog()
                if dialog is not None:
                    if ch in ("y", "Y"):
                        warnings = _execute_delete(dialog)
                        ui.add_warnings(warnings)
                        ui.clear_multi()
                        ui.close_dialog()
                    elif ch in ("n", "N", "\x1b", "\x03"):
                        ui.close_dialog()
                    continue

                if ch in ("q", "Q", "\x03"):
                    ui.quit()
                    stop.set()
                    return
                if ch == "\x1b":
                    r2, _, _ = select.select([fd], [], [], 0.05)
                    if not r2:
                        continue
                    if os.read(fd, 1) != b"[":
                        continue
                    r3, _, _ = select.select([fd], [], [], 0.05)
                    if not r3:
                        continue
                    ch3 = os.read(fd, 1).decode("utf-8", errors="replace")
                    tree, rows = build()
                    if not rows:
                        continue
                    sel = ui.selected
                    cur = rows[sel].node
                    if ch3 == "A":
                        ui.move(-1)
                    elif ch3 == "B":
                        ui.move(1)
                    elif ch3 == "C":
                        ui.toggle_detail()
                    elif ch3 == "D":
                        if ui.detail_open():
                            ui.close_detail()
                        elif ui.is_expanded(cur.node_id):
                            ui.collapse(cur.node_id)
                    continue

                if ch in ("\r", "\n"):
                    tree, rows = build()
                    if rows:
                        ui.toggle_expand(rows[ui.selected].node.node_id)
                elif ch == " " or ch == "x":
                    tree, rows = build()
                    if rows:
                        ui.toggle_multi(rows[ui.selected].node.node_id)
                elif ch == "a":
                    tree, rows = build()
                    ui.select_all_visible(rows)
                elif ch == "A":
                    ui.clear_multi()
                elif ch == "d":
                    tree, rows = build()
                    cur = rows[ui.selected].node if rows else None
                    targets = _collect_targets(tree, ui, cur)
                    if targets:
                        ui.open_dialog(_prepare_dialog(targets))
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)


__all__ = ["TreeMonitor"]
