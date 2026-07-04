"""Tests for ``materialize_plan_records`` — surfacing a plan on Agents + Knowledge.

After a PlanMode run, the plan appears as an agent-task session (Agents tab) and
as a **sourced ``KnowledgeItem``** (integration.md §5): auto-derived knowledge is
typed + source-attributed (≥1 ``SourceRef`` — invariant #4), not an unsourced
free-form ``Note``. It surfaces on the coordination-loop knowledge read-model
(``assemble_workspace_context(...).knowledge``), not the Note-docs Knowledge tab.
"""

from __future__ import annotations

from pathlib import Path

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.services.agent_task_store import (
    list_agent_task_metadata,
    read_agent_task_events,
)
from molexp.services.plan_runtime import materialize_plan_records
from molexp.workspace import Bundle
from molexp.workspace.knowledge_item import KnowledgeItem


def _find_record(workspace, run_id: str) -> KnowledgeItem | None:
    """The plan-record ``KnowledgeItem`` for *run_id*, walked from the Bundle."""
    bundle = Bundle(workspace.root)
    return next(
        (c for c in bundle.walk() if isinstance(c, KnowledgeItem) and run_id in c.name),
        None,
    )


def _seed_report(run) -> None:
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    store.put_json(
        "experiment_report",
        {"title": "Melt plan", "objective": "measure conductivity", "assumptions": ["a", "b"]},
        created_by="test",
        parent_ids=[],
    )


def test_materialize_plan_records_writes_agent_session(workspace, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec1")
    _seed_report(run)

    materialize_plan_records(
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


def test_plan_record_is_sourced_knowledge_item(workspace, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec2")
    _seed_report(run)

    materialize_plan_records(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec2",
        draft="build a melt",
        model="m",
    )

    rec = _find_record(workspace, "planrec2")
    assert rec is not None, "the plan record must be a KnowledgeItem"
    # Source attribution invariant (#4): ≥1 SourceRef, citing the run + experiment.
    meta = rec.read_knowledge_meta()
    assert len(meta.sources) >= 1
    kinds = {(s.kind, s.ref) for s in meta.sources}
    assert ("run", run.id) in kinds
    assert ("experiment", experiment.id) in kinds
    # The rendered report is the readable body.
    body = rec.body()
    assert "Melt plan" in body
    assert "measure conductivity" in body
    assert "build a melt" in body  # the original request
    # A typed provenance edge makes the item reachable from the run it cites.
    assert any(e.target for e in rec.typed_out_edges())


def test_materialize_plan_records_writes_session_transcript(workspace, experiment):
    # The plan session carries a synthesized transcript: a step per pipeline
    # stage + a final answer with the spec/workflow — so the session view is the
    # single home for the whole flow.
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrec4")
    _seed_report(run)

    materialize_plan_records(
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


def test_materialize_plan_records_no_report_still_writes_session(workspace, experiment):
    # A plan run with no experiment_report still lists as a session (no knowledge item).
    run = experiment.add_run(params={"mode": "plan", "draft": "x"}, id="planrec3")

    materialize_plan_records(
        run=run,
        experiment=experiment,
        workspace_root=str(workspace.root),
        task_id="plan-planrec3",
        draft="x",
        model="m",
    )

    tasks = list_agent_task_metadata(str(workspace.root))
    assert any(t.task_id == "plan-planrec3" for t in tasks)
    assert _find_record(workspace, "planrec3") is None
