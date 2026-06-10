"""Rich rendering for the ``molexp explore`` TUI.

Pure presentation: every function here maps tree-model state (plus the
small dialog dataclass) to a Rich renderable. No terminal input, no
threads, no workspace mutation — those live in
:mod:`molexp.cli.tui.tree_monitor`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from molexp._run_display import read_run_json
from molexp._typing import JSONValue
from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Experiment, Project, Run, Workspace

from .tree_model import (
    NodePath,
    TreeNode,
    VisibleRow,
    _as_dict,
    _as_str,
    _as_str_dict,
    node_path_str,
)


@dataclass
class _DeleteDialog:
    """Pending confirm dialog describing targets and cancel outcomes."""

    targets: list[TreeNode]
    plan_lines: list[tuple[str, str, str]]  # (state_icon, label, note)


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
    data = read_run_json(run.run_dir)
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
    data = read_run_json(run.run_dir)
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
        named = sorted(p.name for p in Path(logs_dir).glob("*.log"))
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
