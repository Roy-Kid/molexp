"""``ReviewMode`` — read-only typed review, driven on the harness.

ReviewMode is the fifth mode (Chat / Plan / Author / Run / Review). It
reviews an *existing* artefact — a typed
:class:`~molexp.agent.modes._planning.PlanGraph`, a materialized
workspace, or a completed run — against the shared typed contracts. The
:class:`~molexp.agent.modes._planning.IntentSpec` is the contract every
artefact is judged against, so the review never forgets the user's
original goal.

ReviewMode is **strictly read-only**: it ingests the target, runs the
three pure checker functions in
:mod:`~molexp.agent.modes.review.checks`, folds the findings into a
typed :class:`~molexp.agent.modes.review.verdict.ReviewVerdict`, and
persists that verdict through its own
:class:`~molexp.agent.modes.review.verdict_folder.ReviewVerdictFolder`.
It never edits code, never re-runs a workflow, and writes nowhere else.

After ``agent-mode-stage-pipeline-03``, the three stages live as
first-class :class:`~molexp.agent.modes.review.stages.IngestReviewTarget`
/ :class:`RunReviewChecks` / :class:`RenderReviewVerdict` Stage
subclasses; :meth:`ReviewMode.run` delegates to
:meth:`AgentMode.run_pipeline`. The terminal
:class:`~molexp.agent.harness.events.ModeCompletedEvent` carries an
:class:`~molexp.agent.mode.AgentRunResult` whose
``mode_state["verdict"]`` holds the JSON-mode :class:`ReviewVerdict`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
)
from molexp.agent.modes.review.stages import (
    IngestReviewTarget,
    RenderReviewVerdict,
    RunReviewChecks,
)
from molexp.agent.modes.review.target import ReviewTarget, ReviewTargetKind
from molexp.agent.modes.review.verdict import ReviewVerdict, StepFinding
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan.plan_folder import PlanFolder

__all__ = ["ReviewMode", "ReviewModeConfig"]


class ReviewModeConfig(BaseModel):
    """Tunables for :class:`ReviewMode`.

    Attributes:
        verdict_folder_name: Name of the
            :class:`~molexp.agent.modes.review.verdict_folder.ReviewVerdictFolder`
            ReviewMode mounts under its plan anchor.
        request_llm_summary: When ``True`` (the default) ReviewMode
            asks the router for a one-line natural-language summary; it
            degrades cleanly to a deterministic summary when the router
            cannot complete text.
    """

    model_config = ConfigDict(frozen=True)

    verdict_folder_name: str = "review"
    request_llm_summary: bool = True


_CLASS_PIPELINE = ModePipeline(
    stages=(
        NameOnlyStage("IngestReviewTarget"),
        NameOnlyStage("RunReviewChecks"),
        NameOnlyStage("RenderReviewVerdict"),
    ),
    entry="IngestReviewTarget",
    edges=(
        PipelineEdge(from_stage="IngestReviewTarget", to_stage="RunReviewChecks"),
        PipelineEdge(from_stage="RunReviewChecks", to_stage="RenderReviewVerdict"),
        PipelineEdge(from_stage="RenderReviewVerdict", to_stage="verdict_rendered"),
    ),
    terminal_states=("verdict_rendered",),
)


class ReviewMode(AgentMode):
    """Read-only typed review of an existing artefact — three stages."""

    name = "review"
    pipeline = _CLASS_PIPELINE

    def __init__(
        self,
        *,
        config: ReviewModeConfig | None = None,
        plan_folder: PlanFolder,
        intent: IntentSpec,
        plan_graph: PlanGraph,
        capability_graph: CapabilityGraph,
        workspace_path: Path | None = None,
        run_ref: str | None = None,
    ) -> None:
        """Construct a ReviewMode bound to one reviewable artefact."""
        self.config = config or ReviewModeConfig()
        self.plan_folder = plan_folder
        self._intent = intent
        self._plan_graph = plan_graph
        self._capability_graph = capability_graph
        self._workspace_path = workspace_path
        self._run_ref = run_ref
        # Per-run scratch — stages mutate these, run() reads them in
        # _completion. Reset on every ``run`` call.
        self._target: ReviewTarget | None = None
        self._findings: tuple[StepFinding, ...] = ()
        self._summary: str = ""
        self._verdict: ReviewVerdict | None = None
        self.pipeline = ModePipeline(
            stages=(
                IngestReviewTarget(review_mode=self),
                RunReviewChecks(review_mode=self),
                RenderReviewVerdict(review_mode=self),
            ),
            entry="IngestReviewTarget",
            edges=_CLASS_PIPELINE.edges,
            terminal_states=_CLASS_PIPELINE.terminal_states,
        )

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the three-stage read-only review, yielding events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()
        # Reset per-run state.
        self._target = None
        self._findings = ()
        self._summary = ""
        self._verdict = None

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=user_input,
        ):
            yield event

        yield self._completion(harness)

    def _completion(self, harness: AgentHarness) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        assert self._verdict is not None, "RenderReviewVerdict stage did not run"
        verdict = self._verdict
        breakdown = harness.router.snapshot_usage()
        kind = self._target.kind if self._target is not None else ReviewTargetKind.plan
        text = (
            f"Review of the {kind.value} target finished — "
            f"outcome {verdict.overall} ({len(verdict.findings)} finding(s))."
        )
        harness.session.append_message(Message(role="assistant", content=text))
        mode_state: dict[str, object] = {
            "verdict": verdict.model_dump(mode="json"),
            "overall": verdict.overall,
        }
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))
