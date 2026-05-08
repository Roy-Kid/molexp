"""Workflow IR persistence + cross-process recovery.

When the agent binds a workflow IR via ``experiment.set_workflow(ir)``,
the IR must land on disk at ``<exp_dir>/workflow.json`` and load back
when the experiment is reconstructed by a different process (i.e. the
server is restarted).
"""

from __future__ import annotations

import json

import pytest

from molexp.workflow import WorkflowSpec, default_registry  # noqa: F401  (registers slugs)
from molexp.workspace import Workspace


def _ir() -> dict:
    return {
        "workflow_id": "workflow_00000000",
        "name": "ir_persist",
        "task_configs": [
            {
                "task_id": "k",
                "task_type": "core.constant",
                "config": {"value": 11},
                "status": "pending",
            },
            {
                "task_id": "doubled",
                "task_type": "core.multiply",
                "config": {"factor": 2.0},
                "status": "pending",
            },
        ],
        "links": [
            {"source": "k", "target": "doubled", "mapping": {}, "status": "pending"},
        ],
        "metadata": {"label": None, "description": None, "tags": [], "custom": {}},
    }


class TestPersistence:
    def test_set_workflow_with_ir_writes_file(self, experiment) -> None:
        ir = _ir()
        experiment.set_workflow(ir)

        path = experiment.experiment_dir / "workflow.json"
        assert path.exists()
        with open(path) as fh:
            on_disk = json.load(fh)
        assert on_disk == ir

    def test_workflow_property_lazy_loads_after_restart(self, tmp_path) -> None:
        # Process 1: bind IR
        ws1 = Workspace(root=tmp_path, name="lab")
        proj1 = ws1.project("p")
        exp1 = proj1.experiment("e")
        exp1.set_workflow(_ir())

        # Process 2: same workspace, fresh objects (simulates server restart)
        ws2 = Workspace(root=tmp_path, name="lab")
        proj2 = ws2.project("p")
        # Use list_experiments → reconstruction path
        scanned = proj2.list_experiments()
        exp2 = next(e for e in scanned if e.name == "e")

        assert exp2.workflow is not None
        assert isinstance(exp2.workflow, WorkflowSpec)
        # Same topology hash → same workflow_id
        assert exp2.workflow.workflow_id == exp1.workflow.workflow_id

    @pytest.mark.asyncio
    async def test_recovered_workflow_executes(self, tmp_path) -> None:
        ws1 = Workspace(root=tmp_path, name="lab")
        proj1 = ws1.project("p")
        exp1 = proj1.experiment("e")
        exp1.set_workflow(_ir())

        ws2 = Workspace(root=tmp_path, name="lab")
        proj2 = ws2.project("p")
        exp2 = next(e for e in proj2.list_experiments() if e.name == "e")

        result = await exp2.workflow.execute()
        assert result.outputs["doubled"] == 22.0  # 11 * 2

    def test_double_bind_raises(self, experiment) -> None:
        experiment.set_workflow(_ir())
        with pytest.raises(ValueError, match="already has a workflow"):
            experiment.set_workflow(_ir())

    def test_unknown_slug_in_ir_raises(self, experiment) -> None:
        ir = _ir()
        ir["task_configs"][0]["task_type"] = "no.such.slug"
        with pytest.raises(KeyError, match="no.such.slug"):
            experiment.set_workflow(ir)
        # Nothing should have been persisted
        assert not (experiment.experiment_dir / "workflow.json").exists()
