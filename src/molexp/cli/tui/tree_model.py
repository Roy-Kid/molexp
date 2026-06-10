"""Tree data model for the ``molexp explore`` TUI.

Owns everything the explorer knows about a workspace *before* any
rendering happens: the :class:`TreeNode` dataclass, the tree builder
(workspace → project → experiment → run → execution), status folding,
and the flatten step that turns the tree + an expansion set into the
visible row list. No Rich / terminal imports live here — this module is
the part exercised directly by unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from molexp._run_display import elapsed, read_run_json
from molexp._typing import JSONValue
from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Experiment, Project, Run, Workspace

# Per-node back-pointer used by the delete / detail flows. Each kind of
# tree node carries a different concrete type — workspace nodes hold a
# ``Workspace``, project nodes a ``Project``, etc. Execution nodes pack
# both the run and the execution id into a tuple. The detail / delete
# helpers narrow via ``isinstance`` (or the canonical destructuring for
# the tuple variant).
type TreeNodeRef = Workspace | Project | Experiment | Run | tuple[Run, str] | None

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
    data = read_run_json(run.run_dir)
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
        elapsed=elapsed(_as_str(data.get("created_at")), _as_str(data.get("finished_at"))),
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
    duration = elapsed(_as_str(rec.get("started_at")), _as_str(rec.get("finished_at")))
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
        elapsed=duration,
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


__all__ = [
    "SHORT_ID_LEN",
    "NodeKind",
    "NodePath",
    "TreeNode",
    "TreeNodeRef",
    "VisibleRow",
    "build_tree",
    "flatten",
    "node_path_str",
]
