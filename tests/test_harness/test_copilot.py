"""Workspace Copilot — ``molexp.harness.copilot.summarize_workspace`` (P0.6, integration.md §7.1)."""

from __future__ import annotations

from molexp.harness.copilot import WorkspaceSummary, summarize_workspace
from molexp.workspace import (
    ContextFocus,
    HealthFlag,
    KnowledgeRef,
    RunRef,
    WorkspaceContext,
    WorkspaceRef,
)

_SUMMARY_FIELDS = (
    "workspace",
    "headline",
    "counts",
    "failed_runs",
    "running_runs",
    "health_flags",
    "open_questions",
    "relevant_knowledge",
    "next_actions",
)


def _run(run_id: str, status: str) -> RunRef:
    return RunRef(run_id=run_id, experiment_id="e", project_id="p", status=status)


def _context() -> WorkspaceContext:
    return WorkspaceContext(
        workspace=WorkspaceRef(id="lab", name="Lab", root="/tmp/lab", targets=[]),
        focus=ContextFocus(),
        recent_runs=[_run("r1", "failed"), _run("r2", "running")],
        failed_runs=[_run("r1", "failed")],
        running_runs=[_run("r2", "running")],
        knowledge=[KnowledgeRef(path="k/obs", type="knowledge.item", title="Obs")],
        open_questions=[KnowledgeRef(path="k/oq", type="knowledge.item", title="OQ")],
        stale_or_missing=[
            HealthFlag(kind="failed_run", ref="r1", detail="run r1 is failed (retryable)"),
            HealthFlag(kind="stale_running", ref="r2", detail="run r2 heartbeat stale"),
            HealthFlag(kind="orphan_artifact", ref="a1", detail="asset a1 producer unresolved"),
        ],
    )


class TestSummarizeWorkspace:
    def test_returns_structured_summary_with_all_fields(self) -> None:
        summary = summarize_workspace(_context())
        assert isinstance(summary, WorkspaceSummary)
        for field in _SUMMARY_FIELDS:
            assert hasattr(summary, field), field
        assert summary.workspace.name == "Lab"
        assert "Lab" in summary.headline
        assert summary.counts["failed_runs"] == 1
        assert len(summary.failed_runs) == 1

    def test_next_actions_ranked_most_severe_first(self) -> None:
        """failed → stale → orphan → open-question."""
        actions = summarize_workspace(_context()).next_actions
        kinds = [a.kind for a in actions]

        def _first(prefix_kinds: tuple[str, ...]) -> int:
            return next(i for i, k in enumerate(kinds) if k in prefix_kinds)

        failed_i = _first(("diagnose_failed_run", "retry_failed_run"))
        stale_i = _first(("review_stale_running",))
        orphan_i = _first(("review_orphan_artifact",))
        question_i = _first(("answer_open_question",))
        assert failed_i < stale_i < orphan_i < question_i
        assert actions[0].target == "r1"

    def test_actions_advisory_and_high_risk_requires_proposal(self) -> None:
        """Every action is advisory; only the high-risk retry flags requires_proposal (P0.5)."""
        actions = summarize_workspace(_context()).next_actions
        assert all(a.advisory is True for a in actions)
        by_kind = {a.kind: a for a in actions}
        assert by_kind["retry_failed_run"].requires_proposal is True
        for read_only in (
            "diagnose_failed_run",
            "review_stale_running",
            "review_orphan_artifact",
            "answer_open_question",
        ):
            assert by_kind[read_only].requires_proposal is False

    def test_summarize_is_deterministic_and_read_only(self) -> None:
        ctx = _context()
        assert summarize_workspace(ctx) == summarize_workspace(ctx)
        # ctx is frozen — a second call cannot have mutated it
        assert ctx.failed_runs[0].run_id == "r1"
