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
their cancel path inspected first (see :mod:`molexp._run_cancel`):
cancellable ones are cancelled then deleted; uncancellable ones stay
with a warning printed after the dialog closes.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from molexp._typing import JSONValue
from molexp.workspace import Experiment, Project, Run, Workspace

# Per-node back-pointer used by the delete / detail flows. Each kind of
# tree node carries a different concrete type — workspace nodes hold a
# ``Workspace``, project nodes a ``Project``, etc. Execution nodes pack
# both the run and the execution id into a tuple. The detail / delete
# helpers narrow via ``isinstance`` (or the canonical destructuring for
# the tuple variant).
type TreeNodeRef = Workspace | Project | Experiment | Run | tuple[Run, str] | None

from rich.console import Console, Group, RenderableType  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from molexp._run_cancel import try_cancel  # noqa: E402
from molexp.plugins.submit_molq.metadata import normalize_executor_info  # noqa: E402

if TYPE_CHECKING:
    from molexp.workspace import Workspace
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run


NodeKind = Literal["workspace", "project", "experiment", "run", "execution"]
NodePath = tuple[str, ...]  # e.g. ("project", "proj-a", "experiment", "exp-x")
SHORT_ID_LEN = 6


# ── Tree model ────────────────────────────────────────────────────────────────


@dataclass
class TreeNode:
    """Single node in the workspace tree."""

    kind: NodeKind
    node_id: NodePath
    display_label: str
    status: str | None = None  # pending / running / succeeded / failed / ...
    elapsed: str | None = None
    note: str | None = None  # error message or short info
    count_hint: str | None = None  # e.g. "2 exp, 4 runs"
    children: list[TreeNode] = field(default_factory=list)
    # Opaque reference used by the delete flow.  Never inspected by UI.
    ref: TreeNodeRef = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _as_str(value: JSONValue) -> str | None:
    """Narrow a ``JSONValue`` cell to ``str | None`` for status / timestamp fields."""
    return value if isinstance(value, str) else None


def _as_dict(value: JSONValue) -> dict[str, JSONValue] | None:
    """Narrow a ``JSONValue`` cell to a JSON-shaped dict (or ``None``)."""
    return value if isinstance(value, dict) else None


def _as_str_dict(value: JSONValue) -> dict[str, str] | None:
    """Narrow a ``JSONValue`` cell to ``dict[str, str]`` (label / tag maps)."""
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items() if isinstance(v, str)}


def _short_id(full: str) -> str:
    """Git-style 6-char prefix; leaves short ids untouched."""
    if len(full) <= SHORT_ID_LEN:
        return full
    return full[:SHORT_ID_LEN]


def _short_exec_label(exec_id: str) -> str:
    """Format an execution id: ``exec-<6char>[#N]``."""
    body = exec_id
    if body.startswith("exec-"):
        body = body[5:]
    # Split attempt suffix like "...-2"
    attempt = ""
    if "-" in body:
        parts = body.rsplit("-", 1)
        if parts[1].isdigit():
            body = parts[0]
            attempt = f"#{parts[1]}"
    return f"exec-{_short_id(body)}{attempt}"


def _elapsed(started: str | None, finished: str | None) -> str | None:
    if not started:
        return None
    try:
        s = datetime.fromisoformat(started)
        e = datetime.fromisoformat(finished) if finished else datetime.now()
        secs = max(0, int((e - s).total_seconds()))
        if secs < 60:
            return f"{secs}s"
        m, ss = divmod(secs, 60)
        if m < 60:
            return f"{m}m{ss:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    except Exception:
        return None


def _read_run_json(run_dir: Path) -> dict[str, JSONValue]:
    p = Path(run_dir) / "run.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ── Tree builder ──────────────────────────────────────────────────────────────


def build_tree(
    workspace: Workspace,
    *,
    project_filter: str | None = None,
    experiment_filter: str | None = None,
) -> TreeNode:
    """Walk *workspace* and return a fresh tree.

    Filters apply to the top-level listing: when set they restrict the
    projects/experiments shown at their respective levels.  Children
    always include everything below a surviving node.
    """
    root = TreeNode(
        kind="workspace",
        node_id=("workspace", workspace.name),
        display_label=workspace.name,
        ref=workspace,
    )
    projects: list[Project] = workspace.list_projects()
    projects.sort(key=lambda p: p.id)
    for project in projects:
        if project_filter and project.id != project_filter and project.name != project_filter:
            continue
        root.children.append(_build_project_node(project, experiment_filter))
    _fold_status(root)
    return root


def _build_project_node(project: Project, experiment_filter: str | None) -> TreeNode:
    node = TreeNode(
        kind="project",
        node_id=("project", project.id),
        display_label=project.name,
        ref=project,
    )
    experiments: list[Experiment] = project.list_experiments()
    experiments.sort(key=lambda e: e.id)
    run_count = 0
    for exp in experiments:
        if experiment_filter and exp.id != experiment_filter and exp.name != experiment_filter:
            continue
        child = _build_experiment_node(exp, project.id)
        node.children.append(child)
        run_count += sum(1 for _ in child.children)
    exp_n = len(node.children)
    node.count_hint = f"{exp_n} exp, {run_count} runs"
    return node


def _build_experiment_node(exp: Experiment, project_id: str) -> TreeNode:
    node = TreeNode(
        kind="experiment",
        node_id=("project", project_id, "experiment", exp.id),
        display_label=exp.name,
        ref=exp,
    )
    runs: list[Run] = exp.list_runs()
    runs.sort(key=lambda r: r.id)
    for run in runs:
        node.children.append(_build_run_node(run, project_id, exp.id))
    node.count_hint = f"{len(runs)} runs"
    return node


def _build_run_node(run: Run, project_id: str, exp_id: str) -> TreeNode:
    data = _read_run_json(run.run_dir)
    status_raw = data.get("status")
    status = status_raw if isinstance(status_raw, str) else (run.metadata.status or "pending")
    info = normalize_executor_info(
        _as_dict(data.get("executor_info")), _as_str_dict(data.get("labels"))
    )
    note: str | None = None
    err = data.get("error")
    if isinstance(err, dict):
        note_raw = err.get("message")
        note = note_raw if isinstance(note_raw, str) else None

    node = TreeNode(
        kind="run",
        node_id=("project", project_id, "experiment", exp_id, "run", run.id),
        display_label=_short_id(run.id),
        status=status,
        elapsed=_elapsed(_as_str(data.get("created_at")), _as_str(data.get("finished_at"))),
        note=note,
        ref=run,
    )

    history_raw = data.get("execution_history")
    history = history_raw if isinstance(history_raw, list) else []
    attempts = len(history)
    node.count_hint = f"{attempts} attempts" if attempts else None
    for rec in history:
        if isinstance(rec, dict):
            node.children.append(_build_execution_node(rec, project_id, exp_id, run, info))
    return node


def _build_execution_node(
    rec: dict[str, JSONValue],
    project_id: str,
    exp_id: str,
    run: Run,
    run_executor_info: dict[str, str],
) -> TreeNode:
    exec_id_raw = rec.get("execution_id")
    exec_id = exec_id_raw if isinstance(exec_id_raw, str) else ""
    status_raw = rec.get("status")
    status = status_raw if isinstance(status_raw, str) else "unknown"
    elapsed = _elapsed(_as_str(rec.get("started_at")), _as_str(rec.get("finished_at")))
    sched_raw = rec.get("scheduler_job_id")
    sched = sched_raw if isinstance(sched_raw, str) else run_executor_info.get("scheduler_job_id")
    note = f"sched={sched}" if sched else None
    return TreeNode(
        kind="execution",
        node_id=(
            "project",
            project_id,
            "experiment",
            exp_id,
            "run",
            run.id,
            "execution",
            exec_id,
        ),
        display_label=_short_exec_label(exec_id),
        status=status,
        elapsed=elapsed,
        note=note,
        ref=(run, exec_id),
    )


def _fold_status(node: TreeNode) -> str:
    """Bubble up a summary status string for non-leaf nodes (for breadcrumb)."""
    if not node.children:
        return node.status or ""
    statuses = [_fold_status(c) for c in node.children]
    priority = ("running", "pending", "failed", "cancelled", "succeeded")
    for p in priority:
        if p in statuses:
            node.status = p
            return p
    node.status = statuses[0] if statuses else ""
    return node.status


# ── Flattening (tree → visible rows given expanded set) ───────────────────────


@dataclass
class VisibleRow:
    node: TreeNode
    depth: int
    has_children: bool
    expanded: bool


def flatten(root: TreeNode, expanded: set[NodePath]) -> list[VisibleRow]:
    """Depth-first flatten of *root* respecting the *expanded* set.

    The workspace root itself is rendered in the breadcrumb only — the
    top of the list starts at projects.
    """
    rows: list[VisibleRow] = []

    def walk(n: TreeNode, depth: int) -> None:
        has_children = bool(n.children)
        is_open = n.node_id in expanded
        rows.append(VisibleRow(n, depth, has_children, is_open))
        if is_open:
            for c in n.children:
                walk(c, depth + 1)

    for child in root.children:
        walk(child, 0)
    return rows


def node_path_str(root: TreeNode, target: NodePath) -> str:
    """Return breadcrumb-friendly path ``ws / proj / exp / run / exec``."""

    def walk(n: TreeNode, acc: list[str]) -> list[str] | None:
        if n.node_id == target:
            return [*acc, n.display_label]
        for c in n.children:
            got = walk(c, [*acc, n.display_label])
            if got is not None:
                return got
        return None

    path = walk(root, []) or [root.display_label]
    return "  /  ".join(path)


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


@dataclass
class _DeleteDialog:
    """Pending confirm dialog describing targets and cancel outcomes."""

    targets: list[TreeNode]
    plan_lines: list[tuple[str, str, str]]  # (state_icon, label, note)


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
            from molexp._run_cancel import classify

            plan = classify(node.ref)
            if plan.kind == "none":
                lines.append(("!", _describe_target(node), f"uncancellable: {plan.detail}"))
            else:
                lines.append(("⟳", _describe_target(node), f"cancel via {plan.kind}"))
        elif node.kind == "execution" and running:
            # Executions share cancel path with their parent run.
            assert isinstance(node.ref, tuple) and isinstance(node.ref[0], Run)
            run = node.ref[0]
            from molexp._run_cancel import classify

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


# ── Rendering ─────────────────────────────────────────────────────────────────


_STATE_STYLE = {
    "running": "bold cyan",
    "pending": "yellow",
    "done": "bold green",
    "succeeded": "bold green",
    "failed": "bold red",
    "cancelled": "dim",
    "mixed": "bold yellow",
}


def _state_icon(status: str | None) -> str:
    s = (status or "").lower()
    return {
        "running": "⟳",
        "pending": "·",
        "succeeded": "✓",
        "done": "✓",
        "failed": "✗",
        "cancelled": "—",
    }.get(s, "·")


def _render_breadcrumb(root: TreeNode, rows: list[VisibleRow], selected: int) -> Panel:
    if not rows:
        text = Text(root.display_label, style="bold white")
    else:
        sel_node = rows[selected].node
        text = Text(node_path_str(root, sel_node.node_id), style="bold white")
    return Panel(text, style="dim", padding=(0, 2))


def _render_tree(rows: list[VisibleRow], selected: int, multi: set[NodePath]) -> Panel:
    table = Table(
        show_header=True,
        header_style="dim",
        show_edge=False,
        box=None,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("", width=2)  # select marker
    table.add_column("TREE", ratio=4, no_wrap=False)
    table.add_column("STATE", width=13, no_wrap=True)
    table.add_column("ELAPSED", width=8, no_wrap=True)
    table.add_column("NOTE", ratio=3, overflow="fold")

    if not rows:
        table.add_row("", Text("(empty workspace)", style="dim"), "", "", "")
    for i, row in enumerate(rows):
        n = row.node
        is_sel = i == selected
        is_checked = n.node_id in multi
        if is_checked:
            check_text = Text("■", style="bold magenta")
        else:
            check_text = Text(" ", style="white")

        twist = "▾" if row.expanded else ("▸" if row.has_children else " ")
        indent = "  " * row.depth
        label = n.display_label
        count = f"  [dim]({n.count_hint})[/dim]" if n.count_hint else ""
        tree_cell = Text.from_markup(f"{indent}{twist} {label}{count}")
        if is_sel:
            tree_cell.stylize("on grey19")

        state_style = _STATE_STYLE.get((n.status or "").lower(), "white")
        state_cell = Text(
            f"{_state_icon(n.status)} {(n.status or '').upper()}" if n.status else "",
            style=state_style,
        )

        note = n.note or ""
        table.add_row(
            check_text,
            tree_cell,
            state_cell,
            Text(n.elapsed or "", style="dim"),
            Text(note, style="dim"),
        )

    footer_hint = (
        r"\[↑↓] nav  \[↵] expand  \[→] detail  \[←] back  "
        r"\[space] select  \[a/A] all/clear  \[d] delete  \[q] quit"
    )
    return Panel(
        table,
        title="[bold]Workspace[/bold]",
        subtitle=f"[dim]{footer_hint}[/dim]",
        padding=(0, 1),
    )


def _fmt_iso(value: JSONValue | datetime) -> str:
    if value is None:
        return ""
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except Exception:
        return str(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _kv_table() -> Table:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(overflow="fold")
    return t


def _json_snippet(obj: JSONValue, *, max_lines: int = 12) -> RenderableType:
    try:
        text = json.dumps(obj, indent=2, default=str, sort_keys=True)
    except Exception:
        return Text(str(obj))
    lines = text.splitlines()
    total = len(lines)
    truncated = total > max_lines
    if truncated:
        text = "\n".join(lines[:max_lines])
    syntax = Syntax(
        text,
        "json",
        theme="ansi_dark",
        background_color="default",
        word_wrap=True,
    )
    if truncated:
        return Group(
            syntax,
            Text(f"… (+{total - max_lines} more lines)", style="dim"),
        )
    return syntax


def _detail_workspace(node: TreeNode) -> list[RenderableType]:
    assert isinstance(node.ref, Workspace), "workspace-kind node must hold a Workspace"
    ws = node.ref
    kv = _kv_table()
    kv.add_row("name", str(ws.name))
    kv.add_row("root", str(ws.root))
    if node.count_hint:
        kv.add_row("contents", node.count_hint)
    return [kv]


def _detail_project(node: TreeNode) -> list[RenderableType]:
    assert isinstance(node.ref, Project), "project-kind node must hold a Project"
    proj = node.ref
    meta = proj.metadata
    kv = _kv_table()
    kv.add_row("id", str(meta.id))
    kv.add_row("name", str(meta.name))
    if meta.description:
        kv.add_row("description", meta.description)
    if meta.tags:
        kv.add_row("tags", ", ".join(meta.tags))
    if meta.owner:
        kv.add_row("owner", meta.owner)
    kv.add_row("created_at", _fmt_iso(meta.created_at))
    if node.count_hint:
        kv.add_row("contents", node.count_hint)
    return [kv]


def _detail_experiment(node: TreeNode) -> list[RenderableType]:
    assert isinstance(node.ref, Experiment), "experiment-kind node must hold an Experiment"
    exp = node.ref
    meta = exp.metadata
    kv = _kv_table()
    kv.add_row("id", str(meta.id))
    kv.add_row("name", str(meta.name))
    if meta.description:
        kv.add_row("description", meta.description)
    if meta.tags:
        kv.add_row("tags", ", ".join(meta.tags))
    wf_src = getattr(meta, "workflow_source", None)
    if wf_src:
        kv.add_row("workflow", str(wf_src))
    n_rep = getattr(meta, "n_replicas", None)
    if n_rep is not None:
        kv.add_row("n_replicas", str(n_rep))
    seeds = getattr(meta, "seeds", None)
    if seeds:
        kv.add_row("seeds", ", ".join(str(s) for s in seeds))
    pspace = getattr(meta, "parameter_space", None) or {}
    if pspace:
        kv.add_row("parameters", _json_snippet(pspace))
    kv.add_row("created_at", _fmt_iso(meta.created_at))
    if node.count_hint:
        kv.add_row("contents", node.count_hint)
    return [kv]


def _detail_run(node: TreeNode) -> list[RenderableType]:
    assert isinstance(node.ref, Run), "run-kind node must hold a Run"
    run = node.ref
    data = _read_run_json(run.run_dir)
    kv = _kv_table()
    kv.add_row("id", str(run.id))
    status_str = _as_str(data.get("status")) or node.status or ""
    kv.add_row("status", status_str.upper())
    if profile := _as_str(data.get("profile")):
        kv.add_row("profile", profile)
    if config_hash := _as_str(data.get("config_hash")):
        kv.add_row("config_hash", config_hash[:16])
    if script := _as_str(data.get("script")):
        kv.add_row("script", script)
    created_at = data.get("created_at")
    if created_at:
        kv.add_row("created_at", _fmt_iso(created_at))
    finished_at = data.get("finished_at")
    if finished_at:
        kv.add_row("finished_at", _fmt_iso(finished_at))
    if node.elapsed:
        kv.add_row("elapsed", node.elapsed)
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if msg:
            kv.add_row("error", str(msg))
    exec_info_dict = _as_dict(data.get("executor_info"))
    if exec_info_dict:
        norm = normalize_executor_info(exec_info_dict, _as_str_dict(data.get("labels")))
        for key in ("backend", "scheduler", "cluster", "job_id", "scheduler_job_id"):
            if norm.get(key):
                kv.add_row(key, str(norm[key]))
    history_raw = data.get("execution_history")
    history = history_raw if isinstance(history_raw, list) else []
    if history:
        kv.add_row("attempts", str(len(history)))
    kv.add_row("run_dir", str(run.run_dir))

    cfg = _as_dict(data.get("config"))
    body: list[RenderableType] = [kv]
    if cfg:
        body.append(Text(""))
        body.append(Text("config:", style="bold cyan"))
        body.append(_json_snippet(cfg, max_lines=18))
    return body


def _detail_execution(node: TreeNode) -> list[RenderableType]:
    assert (
        isinstance(node.ref, tuple)
        and isinstance(node.ref[0], Run)
        and isinstance(node.ref[1], str)
    ), "execution-kind node must hold a (Run, exec_id) tuple"
    run = node.ref[0]
    exec_id = node.ref[1]
    data = _read_run_json(run.run_dir)
    rec: dict[str, JSONValue] | None = None
    history = data.get("execution_history")
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and item.get("execution_id") == exec_id:
                rec = item
                break
    kv = _kv_table()
    kv.add_row("execution_id", str(exec_id))
    kv.add_row("run_id", str(run.id))
    if rec is not None:
        kv.add_row("status", str(rec.get("status", node.status or "")).upper())
        if rec.get("started_at"):
            kv.add_row("started_at", _fmt_iso(rec["started_at"]))
        if rec.get("finished_at"):
            kv.add_row("finished_at", _fmt_iso(rec["finished_at"]))
        if rec.get("scheduler_job_id"):
            kv.add_row("scheduler_job_id", str(rec["scheduler_job_id"]))
    if node.elapsed:
        kv.add_row("elapsed", node.elapsed)
    exec_dir = run.run_dir / "executions" / exec_id
    kv.add_row("execution_dir", str(exec_dir))
    # Per-attempt artifacts now live under executions/<id>/; surface
    # whichever ones exist so users can locate them at a glance.
    for fname in ("stdout.log", "stderr.log", "workflow.json", "error.txt"):
        candidate = exec_dir / fname
        if candidate.exists():
            kv.add_row(fname, str(candidate))
    logs_dir = exec_dir / "logs"
    if logs_dir.is_dir():
        named = sorted(p.name for p in logs_dir.glob("*.log"))
        if named:
            kv.add_row("logs/", ", ".join(named))
    return [kv]


def _render_detail(node: TreeNode | None) -> Panel:
    if node is None:
        body: list[RenderableType] = [Text("(select a node to see details)", style="dim")]
        title = "Details"
    else:
        try:
            if node.kind == "workspace":
                body = _detail_workspace(node)
            elif node.kind == "project":
                body = _detail_project(node)
            elif node.kind == "experiment":
                body = _detail_experiment(node)
            elif node.kind == "run":
                body = _detail_run(node)
            elif node.kind == "execution":
                body = _detail_execution(node)
            else:
                body = [Text(str(node.display_label))]
        except Exception as exc:  # pragma: no cover - defensive
            body = [Text(f"(failed to load details: {exc})", style="red")]
        title = f"[bold]{node.kind.capitalize()} · {node.display_label}[/bold]"
    return Panel(
        Group(*body),
        title=title,
        padding=(1, 2),
        border_style="cyan",
    )


def _render_dialog(dialog: _DeleteDialog) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=2)
    grid.add_column(ratio=2)
    grid.add_column(ratio=3)
    for icon, label, note in dialog.plan_lines:
        icon_style = {"!": "bold red", "⟳": "bold cyan"}.get(icon, "green")
        grid.add_row(
            Text(icon, style=icon_style),
            Text(label, style="white"),
            Text(note, style="dim"),
        )
    summary = Text(
        f"Delete {len(dialog.targets)} item(s)?  ",
        style="bold yellow",
    )
    hint = Text(
        "[y] confirm    [n / Esc] cancel    Running items marked ! will be SKIPPED (not deleted).",
        style="dim",
    )
    return Panel(
        Group(summary, grid, Text(""), hint),
        title="[bold red]Confirm delete[/bold red]",
        padding=(1, 3),
        border_style="red",
    )


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


__all__ = [
    "TreeMonitor",
    "TreeNode",
    "VisibleRow",
    "build_tree",
    "flatten",
    "node_path_str",
]
