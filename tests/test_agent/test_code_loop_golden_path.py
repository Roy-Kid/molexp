"""Locks the agent code-loop golden path (agent-code-loop-01-golden).

Pure Python API only — no agent loop, no MCP, no UI. The example under
``examples/agent/code_loop_golden_path.py`` is both the human recipe and
the importable implementation under test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from examples.agent.code_loop_golden_path import (
    EXPERIMENT_NAME,
    FIGURE_REL,
    PARAM_AXIS,
    PARAM_VALUES,
    PROJECT_NAME,
    TASK_NAME,
    maybe_plot_records,
    run_code_loop_golden_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "agent" / "code_loop_golden_path.py"


def test_creates_project_and_experiment_layout(tmp_path: Path) -> None:
    """AC-001: projects/<slug>/ + experiments/<slug>/ after scaffold."""
    root = tmp_path / "ws"
    run_code_loop_golden_path(root, plot=False)

    project_dir = root / "projects" / PROJECT_NAME
    experiment_dir = project_dir / "experiments" / EXPERIMENT_NAME
    assert project_dir.is_dir()
    assert experiment_dir.is_dir()
    assert (project_dir / "project.json").is_file()
    assert (experiment_dir / "experiment.json").is_file()


def test_grid_sweep_executes_and_yields_succeeded_records(tmp_path: Path) -> None:
    """AC-002: one succeeded record per grid point with task output key."""
    records = run_code_loop_golden_path(tmp_path / "ws", plot=False)

    assert len(records) == len(PARAM_VALUES)
    assert all(rec["status"] == "succeeded" for rec in records)
    by_x = {rec[PARAM_AXIS]: rec for rec in records}
    for x in PARAM_VALUES:
        assert by_x[x][TASK_NAME] == pytest.approx(x * x)
        assert by_x[x]["run_id"]


def test_scaffold_and_sweep_are_idempotent(tmp_path: Path) -> None:
    """Re-add project/experiment and re-declare the same sweep: same run count."""
    root = tmp_path / "ws"
    first = run_code_loop_golden_path(root, plot=False)
    second = run_code_loop_golden_path(root, plot=False)
    assert len(second) == len(first) == len(PARAM_VALUES)
    assert {r["run_id"] for r in first} == {r["run_id"] for r in second}


def test_example_script_entrypoint_prints_records() -> None:
    """AC-003: standalone ``python examples/agent/code_loop_golden_path.py``."""
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "status=succeeded" in completed.stdout
    assert TASK_NAME in completed.stdout


def test_molplot_branch_soft_skips_or_writes_figure(tmp_path: Path) -> None:
    """AC-004: missing molplot → soft skip; installed → figure under workspace."""
    root = tmp_path / "ws"
    records = run_code_loop_golden_path(root, plot=False)

    molplot = pytest.importorskip("molplot")
    assert molplot is not None

    fig_path = maybe_plot_records(records, root / FIGURE_REL)
    assert fig_path is not None
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_maybe_plot_returns_none_without_records(tmp_path: Path) -> None:
    assert maybe_plot_records([], tmp_path / "out.png") is None
