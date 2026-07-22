"""``persist_plan_workflow_to_experiment`` — plan→experiment provenance stamp (vision-loop-10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Workspace


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    return ws


class TestPersistPlanWorkflowToExperiment:
    def test_stamps_plan_run_id_and_survives_reload(self, workspace: Workspace) -> None:
        """The experiment records WHICH plan run generated its workflow, durably."""
        from molexp.harness.store.file_artifact_store import FileArtifactStore
        from molexp.services.plan_runtime.persist import persist_plan_workflow_to_experiment

        experiment = workspace.add_project("p").add_experiment("e")
        run = experiment.add_run(params={"mode": "plan", "draft": "d"}, id="provenance1")
        store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
        source = (
            "from molexp.workflow import TaskContext, WorkflowCompiler\n\n"
            "def build_workflow():\n"
            "    wf = WorkflowCompiler(name='w')\n"
            "    @wf.task\n"
            "    async def t(ctx: TaskContext) -> dict:\n"
            "        return {'ok': 1}\n"
            "    return wf\n"
        )
        store.put_json(
            "workflow_source",
            {"source": source, "module_name": "generated_workflow", "bound_workflow_id": "bw"},
            created_by="test",
            parent_ids=[],
        )

        assert persist_plan_workflow_to_experiment(run, experiment) is True
        assert experiment.metadata.plan_run_id == "provenance1"
        # The stamp survives a fresh load of the entity file.
        reloaded = Workspace(workspace.resolve()).get_project("p").get_experiment("e")
        assert reloaded.metadata.plan_run_id == "provenance1"
