"""``materialize_plan_records`` — honest outcome + plan回流补全 (vision-loop-06 §A/§B).

Pins the record layer's honesty contract:

- the return value is a structured :class:`PlanRecordOutcome` naming every
  record written and every record that failed — nothing warning-swallowed,
  one broken writer never aborts its siblings;
- the execute tail's ``final_report`` becomes a ``Finding`` (artifact-level
  SourceRefs + a ``references`` edge back to the Decision record);
- a terminal plan failure becomes a ``FailureAnalysis`` + a *visible* failed
  Agents-tab entry (suspensions never do — CLI caller tests below);
- records mount under their EXPERIMENT (the record.py root-mount dodge is
  gone) and every written KnowledgeMeta carries a timestamp.

Offline + deterministic: canned artifacts via ``FileArtifactStore``; the CLI
caller tests stub the gateway/preflight seams exactly like
``tests/test_cli/test_plan_cmd.py`` (that file is unmodified).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from molexp.services.agent_task_store import list_agent_task_metadata
from molexp.services.plan_runtime import (
    PlanFailure,
    PlanRecordError,
    PlanRecordOutcome,
    materialize_plan_records,
)
from molexp.workspace import Bundle, Workspace
from molexp.workspace.knowledge_item import KNOWLEDGE_ITEM_KIND, KnowledgeItem

_DRAFT = "Simulate NEMD ionic mobility"
_MODEL = "stub-model"

_EXPERIMENT_REPORT = {
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "assumptions": ["a", "b"],
}
_FINAL_REPORT = {
    "title": "Water NEMD outcome",
    "conclusion": "Mobility rises with the applied field strength.",
    "hypothesis_verdict": "supported",
    "metrics": {"mobility": 0.42},
}

# Compiles through the public workflow API — persist_plan_workflow_to_experiment
# execs it for the UI graph, so the clean-outcome test gets workflow_ir=written.
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    return wf
"""
_WORKFLOW_SOURCE = {
    "source": _VALID_SOURCE,
    "module_name": "water_nemd",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler", "TaskContext"],
}


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path / "lab", name="lab")


@pytest.fixture()
def experiment(workspace: Workspace) -> Any:
    return workspace.add_project("demo").add_experiment("nemd")


@pytest.fixture()
def run(experiment: Any) -> Any:
    return experiment.add_run(params={"mode": "plan", "draft": _DRAFT}, id="planrec1")


def _seed(run: Any, kind: str, obj: dict[str, Any]) -> Any:
    """Put one canned *kind* artifact into the run's store; returns its ref."""
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    return store.put_json(kind, obj, created_by="test", parent_ids=[])


def _materialize(run: Any, experiment: Any, *, failure: PlanFailure | None = None):
    return materialize_plan_records(
        run=run,
        experiment=experiment,
        workspace_root=str(experiment.project.workspace.root),
        task_id=f"plan-{run.id}",
        draft=_DRAFT,
        model=_MODEL,
        failure=failure,
    )


def _source_pairs(item: KnowledgeItem) -> set[tuple[str, str]]:
    return {(s.kind, s.ref) for s in item.read_knowledge_meta().sources}


# ── honest outcome (§A) ──────────────────────────────────────────────────────


class TestPlanRecordOutcome:
    def test_success_outcome_names_every_written_record_with_no_errors(
        self, run: Any, experiment: Any
    ) -> None:
        """Basics: full success path → structured outcome, all labels, zero errors."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        _seed(run, "workflow_source", _WORKFLOW_SOURCE)

        outcome = _materialize(run, experiment)

        assert isinstance(outcome, PlanRecordOutcome)
        assert outcome.workflow_persisted is True
        assert {"workflow_ir", "agent_task", "session_events", "experiment_record"} <= set(
            outcome.written
        )
        assert outcome.errors == ()

    def test_broken_writer_lands_in_errors_while_siblings_still_write(
        self, run: Any, experiment: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """De-swallow: one raising writer is COLLECTED (never swallowed, never
        aborting) — the sibling records still land on disk."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)

        def _boom(**kwargs: Any) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr("molexp.services.plan_runtime.record.write_agent_task_record", _boom)

        outcome = _materialize(run, experiment)

        broken = [e for e in outcome.errors if e.record == "agent_task"]
        assert broken, outcome.errors
        assert isinstance(broken[0], PlanRecordError)
        assert "disk full" in broken[0].error
        # Siblings were not aborted:
        assert "session_events" in outcome.written
        assert "experiment_record" in outcome.written
        assert experiment.has_folder(
            f"experiment-record-{experiment.id}-{run.id}", cls=KnowledgeItem
        )


# ── mount flip + timestamp (§B) ──────────────────────────────────────────────


class TestRecordMountAndTimestamp:
    def test_decision_record_mounts_under_the_experiment_not_the_root(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """The record.py root-mount dodge is gone: the Decision item is a
        direct child of its experiment, with no doubled path segments."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)

        _materialize(run, experiment)

        name = f"experiment-record-{experiment.id}-{run.id}"
        item = experiment.get_folder(name, cls=KnowledgeItem)
        item_dir = Path(item.resolve()).resolve()
        assert item_dir.parent == Path(experiment.experiment_dir).resolve()
        parts = item_dir.relative_to(Path(workspace.root).resolve()).parts
        assert all(a != b for a, b in pairwise(parts)), parts
        assert not (Path(workspace.root) / name).exists(), "root-mount dodge must be deleted"

    def test_every_written_knowledge_meta_carries_a_timestamp(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """Timestamp stamping — the recency signal the slice-05 digest sorts on."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        _seed(run, "final_report", _FINAL_REPORT)

        _materialize(run, experiment)

        items = [c for c in Bundle(workspace.root).walk() if isinstance(c, KnowledgeItem)]
        assert len(items) >= 2, [i.name for i in items]  # Decision + Finding
        for item in items:
            assert item.read_knowledge_meta().timestamp is not None, item.name


# ── event spine (vision-loop-12): the record write is a knowledge.created emit ─


class TestExperimentRecordEventEmit:
    def test_writing_the_experiment_record_emits_knowledge_created(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """The Decision-record write lands exactly one ``knowledge.created`` on
        the workspace event spine: actor ``plan-record``, ref = the item's
        workspace-relative path, payload ``{type, title}``."""
        from molexp.workspace.events import read_workspace_events

        _seed(run, "experiment_report", _EXPERIMENT_REPORT)

        _materialize(run, experiment)

        events = read_workspace_events(workspace.root, type="knowledge.created")
        assert len(events) == 1
        event = events[0]
        assert event.actor == "plan-record"
        item = experiment.get_folder(
            f"experiment-record-{experiment.id}-{run.id}", cls=KnowledgeItem
        )
        rel_path = (
            Path(item.resolve()).resolve().relative_to(Path(workspace.root).resolve()).as_posix()
        )
        assert event.refs == [rel_path]
        assert event.payload["type"] == KNOWLEDGE_ITEM_KIND
        assert event.payload["title"] == "Water NEMD"


# ── Finding (execute-tail success, §B) ───────────────────────────────────────


class TestFindingRecord:
    def test_finding_written_when_final_report_artifact_present(
        self, run: Any, experiment: Any
    ) -> None:
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        final_ref = _seed(run, "final_report", _FINAL_REPORT)

        outcome = _materialize(run, experiment)

        assert "finding" in outcome.written
        item = experiment.get_folder(f"finding-{experiment.id}-{run.id}", cls=KnowledgeItem)
        assert item.read_knowledge_meta().kind == "Finding"
        pairs = _source_pairs(item)
        assert ("run", run.id) in pairs
        assert ("experiment", experiment.id) in pairs
        assert ("artifact", final_ref.id) in pairs, "the final_report ref must be cited"
        assert "Mobility rises" in item.body()

    def test_finding_cites_the_run_and_references_the_decision_record(
        self, run: Any, experiment: Any
    ) -> None:
        """Integration: plan → outcome stays traversable — derived_from the run
        plus a references edge to the Decision record."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        _seed(run, "final_report", _FINAL_REPORT)

        _materialize(run, experiment)

        item = experiment.get_folder(f"finding-{experiment.id}-{run.id}", cls=KnowledgeItem)
        edges = item.typed_out_edges()
        assert any(
            e.role == "derived_from" and str(e.target).endswith(f"run-{run.id}") for e in edges
        ), edges
        decision_name = f"experiment-record-{experiment.id}-{run.id}"
        assert any(
            e.role == "references" and str(e.target).endswith(decision_name) for e in edges
        ), edges

    def test_no_finding_when_final_report_absent(self, run: Any, experiment: Any) -> None:
        """Edge case: a plan that never ran the execute tail harvests nothing."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)

        outcome = _materialize(run, experiment)

        assert "finding" not in outcome.written
        assert not experiment.has_folder(f"finding-{experiment.id}-{run.id}", cls=KnowledgeItem)


# ── FailureAnalysis (terminal plan failure, §B) ──────────────────────────────


class TestFailureAnalysisRecord:
    def test_failure_materializes_a_failure_analysis_item(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """failure=PlanFailure(...) → a FailureAnalysis under the experiment:
        failed stage + error + completed artifact kinds + the resume hint."""
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)

        outcome = _materialize(
            run,
            experiment,
            failure=PlanFailure(stage="generate_experiment_spec", error="boom exploded"),
        )

        assert "failure_analysis" in outcome.written
        item = experiment.get_folder(f"failure-{experiment.id}-{run.id}", cls=KnowledgeItem)
        meta = item.read_knowledge_meta()
        assert meta.kind == "FailureAnalysis"
        assert meta.timestamp is not None
        pairs = _source_pairs(item)
        assert ("run", run.id) in pairs
        assert ("experiment", experiment.id) in pairs
        body = item.body()
        assert "generate_experiment_spec" in body
        assert "boom exploded" in body
        assert "experiment_report" in body, "completed artifact kinds must be listed"
        assert "resume" in body.lower(), "the resume hint must be present"

    def test_failure_writes_a_failed_agents_tab_entry(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """A failed plan finally shows in the Agents tab — status ``failed``."""
        _materialize(run, experiment, failure=PlanFailure(stage=None, error="boom"))

        entries = {t.task_id: t for t in list_agent_task_metadata(str(workspace.root))}
        entry = entries.get(f"plan-{run.id}")
        assert entry is not None, entries
        assert entry.status == "failed"
        assert entry.plan_mode is True
        assert entry.project_id == experiment.project.id
        assert entry.experiment_id == experiment.id
        assert entry.run_id == run.id

    def test_failure_session_transcript_carries_error_reason(
        self, workspace: Workspace, run: Any, experiment: Any
    ) -> None:
        """Agents chat must surface the failure reason, not a green 'plan ready'."""
        from molexp.services.agent_task_store import read_agent_task_events

        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        _materialize(
            run,
            experiment,
            failure=PlanFailure(stage="execute_tests", error="pytest exit 2 — Frame missing"),
        )

        events = read_agent_task_events(str(workspace.root), f"plan-{run.id}")
        types = [e["type"] for e in events]
        assert "error" in types, events
        err = next(e for e in events if e["type"] == "error")
        assert "Frame missing" in err["payload"]["message"]
        assert err["payload"]["stage"] == "execute_tests"
        completed = [e for e in events if e["type"] == "loop_completed"]
        assert completed, events
        last = completed[-1]
        assert last["payload"].get("failed") is True
        assert "Frame missing" in last["payload"]["text"]
        assert "plan ready" not in last["payload"]["text"].lower()

    def test_failure_path_writes_no_decision_or_finding(self, run: Any, experiment: Any) -> None:
        """Failure materialization never writes Decision/Finding (success-only).

        Workflow IR is partial product of earlier stages — when the artifact
        exists it still lands on the experiment so the UI graph is not blank.
        """
        _seed(run, "experiment_report", _EXPERIMENT_REPORT)
        _seed(run, "final_report", _FINAL_REPORT)
        _seed(run, "workflow_source", _WORKFLOW_SOURCE)

        outcome = _materialize(run, experiment, failure=PlanFailure(stage=None, error="boom"))

        assert outcome.workflow_persisted is True
        assert "workflow_ir" in outcome.written
        assert "experiment_record" not in outcome.written
        assert "finding" not in outcome.written
        # Bound for UI: in-memory IR and/or externalized workflow.json on save.
        from molexp.workspace.experiment import WORKFLOW_DOC_FILENAME

        doc = Path(experiment.experiment_dir) / WORKFLOW_DOC_FILENAME
        assert experiment.metadata.workflow_source is not None or doc.is_file()

    def test_failure_without_workflow_source_skips_ir(self, run: Any, experiment: Any) -> None:
        """No workflow_source artifact → failure path does not invent an IR error."""
        outcome = _materialize(run, experiment, failure=PlanFailure(stage=None, error="early boom"))

        assert outcome.workflow_persisted is False
        assert "workflow_ir" not in outcome.written
        assert not any(e.record == "workflow_ir" for e in outcome.errors)


# ── CLI caller (failure materializes; suspension never does) ─────────────────


def _patch_cli_stubs(
    monkeypatch: pytest.MonkeyPatch, agents: dict[str, tuple[dict[str, Any], str]]
) -> None:
    """Stub the ``molexp plan`` seams (offline, no LLM) with *agents* canned.

    Mirrors ``tests/test_cli/test_plan_cmd.py`` idioms without touching it. An
    agent the pipeline calls that is NOT in *agents* raises inside its stage —
    the deterministic terminal failure the failure-path test relies on.
    """

    def _fake_preflight(*, model: str) -> Any:
        return None

    def _fake_gateway(*, model: str, run: Any, router: Any = None, **_kwargs: Any) -> Any:
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        gw = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
        for name, (output, kind) in agents.items():
            gw.register(name, output, output_kind=kind)
        return gw

    def _fake_executor() -> Any:
        from molexp.harness import DryRunExecutor

        return DryRunExecutor()

    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.preflight", _fake_preflight)
    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.build_gateway", _fake_gateway)
    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.build_executor", _fake_executor)
    monkeypatch.setattr("molexp.cli.plan_cmd._resolve_grounding", lambda *_a, **_k: None)


_EXPERIMENT_SPEC = {
    "id": "spec-water",
    "experiment_report_id": "rep-water",
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "variables": [],
    "controlled_conditions": [],
    "resolved_questions": [],
    "assumptions": [],
}


class TestCliCallerMaterialization:
    """``molexp plan`` — StageExecutionError materializes ``failure=``;
    an ApprovalPendingError suspension (exit 2) materializes NOTHING."""

    @pytest.mark.integration
    def test_cli_failure_path_materializes_failure_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plan that terminally fails leaves a FailureAnalysis + a failed
        Agents-tab entry on disk before exiting 1."""
        from typer.testing import CliRunner

        from molexp.cli import app
        from molexp.cli._common import deterministic_run_id

        # Only the report writer is canned → GenerateExperimentSpec fails.
        _patch_cli_stubs(
            monkeypatch, {"experiment_report_writer": (_EXPERIMENT_REPORT, "experiment_report")}
        )
        draft = "Simulate NEMD ionic mobility"
        result = CliRunner().invoke(
            app,
            [
                "plan",
                draft,
                "--workspace",
                str(tmp_path),
                "--model",
                _MODEL,
                "--project",
                "demo",
                "--experiment",
                "nemd",
                "--yes",
            ],
        )

        assert result.exit_code == 1, result.output
        run_id = deterministic_run_id({"mode": "plan", "draft": draft})
        ws = Workspace(tmp_path)
        exp = ws.get_project("demo").get_experiment("nemd")
        item = exp.get_folder(f"failure-nemd-{run_id}", cls=KnowledgeItem)
        assert item.read_knowledge_meta().kind == "FailureAnalysis"
        entries = {t.task_id: t for t in list_agent_task_metadata(str(tmp_path))}
        entry = entries.get(f"plan-{run_id}")
        assert entry is not None, entries
        assert entry.status == "failed"

    @pytest.mark.integration
    def test_cli_approval_suspension_materializes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ApprovalPendingError is a suspension, not a failure: exit 2 with NO
        FailureAnalysis and NO failed agent-task entry — a plan that later
        resumes and succeeds must not leave a phantom failure behind."""
        from typer.testing import CliRunner

        from molexp.cli import app
        from molexp.cli._common import deterministic_run_id

        # Both pre-gate agents canned → the run reaches the spec gate and,
        # without --yes on a non-TTY, SUSPENDS there.
        _patch_cli_stubs(
            monkeypatch,
            {
                "experiment_report_writer": (_EXPERIMENT_REPORT, "experiment_report"),
                "experiment_spec_generator": (_EXPERIMENT_SPEC, "experiment_spec"),
            },
        )
        draft = "Simulate NEMD ionic mobility"
        result = CliRunner().invoke(
            app,
            [
                "plan",
                draft,
                "--workspace",
                str(tmp_path),
                "--model",
                _MODEL,
                "--project",
                "demo",
                "--experiment",
                "nemd",
            ],
        )

        assert result.exit_code == 2, result.output
        run_id = deterministic_run_id({"mode": "plan", "draft": draft})
        ws = Workspace(tmp_path)
        failures = [
            c
            for c in Bundle(ws.root).walk()
            if isinstance(c, KnowledgeItem) and c.read_knowledge_meta().kind == "FailureAnalysis"
        ]
        assert failures == [], "a suspension must not materialize a FailureAnalysis"
        assert all(
            t.task_id != f"plan-{run_id}" for t in list_agent_task_metadata(str(tmp_path))
        ), "a suspension must not write a failed agent-task entry"
