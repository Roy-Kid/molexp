"""``Experiment.add_runs`` — materialize a ParamSpace into content-addressed Runs.

Run-materialization semantics owned here:
- one Run per Cartesian cell, each carrying exactly that cell's params;
- run ids are content-addressed via ``derive_run_id`` and stay stable across a
  re-opened Experiment (second-process simulation);
- re-materializing the same space is idempotent (no dupes, no ``RunExistsError``);
- an empty space yields ``[]``.

``derive_run_id``'s own properties (determinism, order-invariance, hex shape)
are owned by ``test_derive_run_id.py`` — this file only pins that ``add_runs``
wires ids through it.
"""

from __future__ import annotations

from molexp.workspace import GridSpace, Workspace
from molexp.workspace.utils import derive_run_id


def _grid() -> GridSpace:
    return GridSpace({"lr": [1e-4, 5e-4], "batch": [32, 64]})


class TestAddRuns:
    def test_materializes_one_run_per_cell_with_cell_params(self, experiment) -> None:
        expected_cells = [dict(cell) for cell in _grid()]
        runs = experiment.add_runs(_grid())

        assert len(experiment.list_runs()) == len(expected_cells) == 4
        materialized = [dict(r.parameters) for r in runs]
        assert len(materialized) == len(expected_cells)
        for cell in expected_cells:
            assert cell in materialized

    def test_derives_content_addressed_id_from_params(self, experiment) -> None:
        for run in experiment.add_runs(_grid()):
            assert run.id == derive_run_id(dict(run.parameters))

    def test_ids_stable_across_reopened_experiment(self, workspace, tmp_path) -> None:
        project = workspace.add_project("det-project")
        experiment = project.add_experiment("det-exp", params={"lr": 1e-4})
        first_ids = {r.id for r in experiment.add_runs(_grid())}

        # Second process: a fresh Workspace over the same root.
        reopened = Workspace(root=tmp_path).get_project("det-project").get_experiment("det-exp")
        assert {r.id for r in reopened.list_runs()} == first_ids

    def test_is_idempotent_on_repeated_materialization(self, experiment) -> None:
        experiment.add_runs(_grid())
        experiment.add_runs(_grid())
        assert len(experiment.list_runs()) == 4

    def test_empty_space_returns_empty_list(self, experiment) -> None:
        assert experiment.add_runs(GridSpace({"lr": []})) == []
