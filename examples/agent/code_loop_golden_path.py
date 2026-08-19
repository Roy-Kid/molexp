"""Agent code-loop golden path — pure Python API, no agent loop.

This is the **recipe** an agent should reimplement by consulting molmcp
(for API discovery / scaffold tools), then writing a script and executing
it. The molexp agent loop does not own create/sweep/plot verbs; it writes
code that calls these public APIs.

Behavior contract (Chat ReAct): see ``DEFAULT_OPS_PREAMBLE`` —
consult molmcp → code_write → code_run → read results; plot with
``import molplot``.

Shape:

1. ``Workspace(...).materialize()``
2. ``add_project`` / ``add_experiment`` (idempotent on slug)
3. Trivial ``WorkflowCompiler`` workflow (stand-in for feature YY)
4. ``exp.sweep(wf, params=...).execute()``
5. ``summary.to_records()`` recovery
6. Optional ``import molplot`` figure under the workspace

Run directly::

    python examples/agent/code_loop_golden_path.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowCompiler
from molexp.workspace.param import GridSpace

# Fixed names so tests can assert the on-disk layout contract.
PROJECT_NAME = "demo"
EXPERIMENT_NAME = "yy-scan"
PARAM_AXIS = "x"
PARAM_VALUES: tuple[float, ...] = (1.0, 2.0, 3.0)
TASK_NAME = "yy_score"
FIGURE_REL = Path("figures") / "yy_scan.png"


def build_yy_workflow() -> WorkflowCompiler:
    """Compile a trivial workflow — stand-in for "implement YY" science."""
    wf = WorkflowCompiler(name="yy")

    @wf.task
    def yy_score(x: float = 0.0) -> float:
        """Root task: sweep param ``x`` binds by name; score is ``x * x``."""
        return x * x

    return wf


def run_code_loop_golden_path(
    workspace_root: Path,
    *,
    plot: bool = True,
) -> list[dict[str, object]]:
    """Execute the golden path under *workspace_root*; return flat records.

    Args:
        workspace_root: Directory that becomes the workspace root (created).
        plot: When True, attempt an optional molplot figure if the package
            is installed; missing molplot is a soft skip (no error).

    Returns:
        One flat ``to_records()`` row per sweep cell (params + task outputs
        + ``run_id`` / ``status`` / ``error``).
    """
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    ws = me.Workspace(root, name="code-loop-golden")
    ws.materialize()
    exp = ws.add_project(PROJECT_NAME).add_experiment(EXPERIMENT_NAME)

    wf = build_yy_workflow()
    space = GridSpace({PARAM_AXIS: list(PARAM_VALUES)})
    summary = exp.sweep(wf, params=space).execute()
    records = summary.to_records()

    if plot:
        maybe_plot_records(records, root / FIGURE_REL)

    return records


def _as_float(value: object) -> float:
    """Coerce a flat record cell to float (params / scalar task outputs)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric record cell, got {type(value).__name__}")
    return float(value)


def maybe_plot_records(
    records: list[dict[str, object]],
    figure_path: Path,
) -> Path | None:
    """Plot ``x`` vs ``yy_score`` with molplot when available.

    Returns:
        The written figure path, or ``None`` when molplot is not installed
        or there are no records to plot.
    """
    if not records:
        return None
    try:
        import molplot
    except ImportError:
        return None

    xs = [_as_float(row[PARAM_AXIS]) for row in records]
    ys = [_as_float(row[TASK_NAME]) for row in records]
    molplot.use("molplot")
    fig, _ax = molplot.line(
        [{"id": TASK_NAME, "x": xs, "y": ys}],
        x_label=PARAM_AXIS,
        y_label=TASK_NAME,
        show_legend=True,
    )
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    return figure_path


def main() -> None:
    workspace_root = Path(tempfile.mkdtemp(prefix="molexp-code-loop-golden-"))
    print(f"workspace root: {workspace_root}")
    records = run_code_loop_golden_path(workspace_root)
    print(f"sweep cells: {len(records)}")
    for row in records:
        print(
            f"  {PARAM_AXIS}={row[PARAM_AXIS]}  status={row['status']}  "
            f"{TASK_NAME}={row[TASK_NAME]}  run_id={row['run_id']}"
        )
    fig = workspace_root / FIGURE_REL
    if fig.is_file():
        print(f"figure: {fig}")
    else:
        print("figure: (skipped — molplot not installed or plot disabled)")


if __name__ == "__main__":
    main()
