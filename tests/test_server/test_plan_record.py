"""Tests for ``record_plan_outputs`` — surfacing a plan on Agents + Knowledge.

After a PlanMode run, the plan should appear as an agent-task session (Agents
tab) and as a Knowledge experiment-record Note, both readable.
"""

from __future__ import annotations

from pathlib import Path

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.server.plan_runtime.record import record_plan_outputs
from molexp.server.routes.agent_task_store import (
    list_agent_task_metadata,
    read_agent_task_events,
)
from molexp.server.routes.knowledge import get_note, list_knowledge


def _seed_report(run) -> None:
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    store.put_json(
        "experiment_report",
        {"title": "Melt plan", "objective": "measure conductivity", "assumptions": ["a", "b"]},
        created_by="test",
        parent_ids=[],
    )


def test_record_plan_outputs_writes_agent_session(workspace, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec1")
    _seed_report(run)

    record_plan_outputs(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec1",
        draft="build a melt",
        model="m",
    )

    tasks = list_agent_task_metadata(str(workspace.root))
    plan_tasks = [t for t in tasks if t.task_id == "plan-planrec1"]
    assert len(plan_tasks) == 1
    assert plan_tasks[0].plan_mode is True
    assert plan_tasks[0].title == "Melt plan"
    assert plan_tasks[0].goal == "build a melt"


def test_record_plan_outputs_writes_readable_knowledge_note(workspace, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec2")
    _seed_report(run)

    record_plan_outputs(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec2",
        draft="build a melt",
        model="m",
    )

    kn = list_knowledge(workspace=workspace)
    note = next((n for n in kn.notes if "planrec2" in n.name), None)
    assert note is not None
    assert note.excerpt  # non-empty — the record renders the report
    # The note opens by its bundle-relative path (root-mounted → no nesting bug).
    detail = get_note(path=note.relPath, workspace=workspace)
    assert "Melt plan" in detail.body
    assert "measure conductivity" in detail.body
    assert "build a melt" in detail.body  # the original request


def test_record_plan_outputs_writes_session_transcript(workspace, experiment):
    # The plan session carries a synthesized transcript: a step per pipeline
    # stage + a final answer with the spec/workflow — so the session view is the
    # single home for the whole flow.
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec4")
    _seed_report(run)

    record_plan_outputs(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec4",
        draft="build a melt",
        model="m",
    )

    events = read_agent_task_events(str(workspace.root), "plan-planrec4")
    types = [e["type"] for e in events]
    assert "tool_call_completed" in types  # at least one stage step
    assert types[-1] == "loop_completed"  # the final answer
    answer = events[-1]["payload"]["text"]
    assert "Melt plan" in answer  # the spec title is in the transcript


def test_record_plan_outputs_no_report_still_writes_session(workspace, experiment):
    # A plan run with no experiment_report still lists as a session (no note).
    run = experiment.add_run(params={"mode": "plan", "draft": "x"}, id="planrec3")

    record_plan_outputs(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec3",
        draft="x",
        model="m",
    )

    tasks = list_agent_task_metadata(str(workspace.root))
    assert any(t.task_id == "plan-planrec3" for t in tasks)
    kn = list_knowledge(workspace=workspace)
    assert not any("planrec3" in n.name for n in kn.notes)
