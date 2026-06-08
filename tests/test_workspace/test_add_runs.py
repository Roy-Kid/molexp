"""Tests for ``Experiment.add_runs`` — materialize a ParamSpace into Runs.

Covers acceptance criteria:
- ac-001: ``add_runs(GridSpace(...))`` mounts one Run per Cartesian cell.
- ac-002: run ids are content-addressed — deterministic from params and
  stable across a re-opened Experiment (second-process simulation).
- ac-003: re-materializing the same space is idempotent (no dupes, no
  ``RunExistsError``, same-id Runs returned).
- edge: empty space yields ``[]``.
"""

from __future__ import annotations

from molexp.workspace import GridSpace, Run, Workspace
from molexp.workspace.utils import derive_run_id


def _grid() -> GridSpace:
    return GridSpace({"lr": [1e-4, 5e-4], "batch": [32, 64]})


# ── ac-001: one Run per Cartesian cell ───────────────────────────────────


def test_add_runs_returns_one_run_per_cell(experiment) -> None:
    """ac-001 — a 2x2 grid yields a list of 4 Run objects."""
    runs = experiment.add_runs(_grid())
    assert isinstance(runs, list)
    assert len(runs) == 4
    assert all(isinstance(r, Run) for r in runs)


def test_add_runs_mounts_each_run_under_experiment(experiment) -> None:
    """ac-001 — each created run is present under the experiment."""
    experiment.add_runs(_grid())
    assert len(experiment.list_runs()) == 4


def test_add_runs_each_run_carries_its_cell_parameters(experiment) -> None:
    """ac-001 — each run's parameters equal exactly one Cartesian cell."""
    expected_cells = [dict(cell) for cell in _grid()]
    runs = experiment.add_runs(_grid())
    materialized = [dict(r.parameters) for r in runs]
    for cell in expected_cells:
        assert cell in materialized
    assert len(materialized) == len(expected_cells)


# ── ac-002: content-addressed, deterministic-across-processes ids ─────────


def test_add_runs_ids_are_derived_from_params(experiment) -> None:
    """ac-002 — each run's id equals derive_run_id of its own params."""
    runs = experiment.add_runs(_grid())
    for r in runs:
        assert r.id == derive_run_id(dict(r.parameters))


def test_add_runs_ids_stable_across_reopened_experiment(workspace, tmp_path) -> None:
    """ac-002 — re-opening the Experiment over the same root yields identical ids."""
    project = workspace.add_project("det-project")
    experiment = project.add_experiment("det-exp", params={"lr": 1e-4})
    first_ids = {r.id for r in experiment.add_runs(_grid())}

    # Simulate a second process: fresh Workspace over the same root.
    reopened_ws = Workspace(root=tmp_path)
    reopened_exp = reopened_ws.get_project("det-project").get_experiment("det-exp")
    reopened_ids = {r.id for r in reopened_exp.list_runs()}

    assert reopened_ids == first_ids


def test_add_runs_ids_are_pure_function_of_params(experiment) -> None:
    """ac-002 — re-deriving ids from the same param dicts matches the run ids."""
    runs = experiment.add_runs(_grid())
    run_ids = {r.id for r in runs}
    rederived = {derive_run_id(dict(cell)) for cell in _grid()}
    assert run_ids == rederived


# ── ac-003: idempotency ──────────────────────────────────────────────────


def test_add_runs_twice_does_not_change_count(experiment) -> None:
    """ac-003 — calling add_runs twice leaves the run count unchanged."""
    experiment.add_runs(_grid())
    assert len(experiment.list_runs()) == 4
    experiment.add_runs(_grid())
    assert len(experiment.list_runs()) == 4


def test_add_runs_twice_returns_same_ids(experiment) -> None:
    """ac-003 — the second call returns the already-existing same-id Runs."""
    first_ids = {r.id for r in experiment.add_runs(_grid())}
    second_ids = {r.id for r in experiment.add_runs(_grid())}
    assert second_ids == first_ids


# ── edge: empty space ────────────────────────────────────────────────────


def test_add_runs_empty_space_returns_empty_list(experiment) -> None:
    """Edge — an empty grid materializes no runs."""
    empty = GridSpace({})
    # An empty GridSpace yields a single empty cell; an axis with no values
    # yields zero cells — use the zero-cell form for a truly empty space.
    empty_axis = GridSpace({"lr": []})
    assert experiment.add_runs(empty_axis) == []
    # Sanity: the no-axis grid is non-empty (one empty-param cell).
    assert len(empty) == 1
