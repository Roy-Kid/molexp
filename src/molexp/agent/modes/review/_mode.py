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

The pipeline is a **plain async stage sequence** on the harness —
``async with harness.stage(name): ...``. There is no
``molexp.workflow`` graph for ReviewMode's own pipeline and no
``pydantic_graph`` import. Three stages:

1. ``IngestReviewTarget`` — detect/load the :class:`ReviewTarget`.
2. ``RunReviewChecks`` — run the three checkers, collect findings,
   optionally ask the router for a one-line summary (degrades cleanly).
3. ``RenderReviewVerdict`` — fold into a :class:`ReviewVerdict` and
   persist it through the :class:`ReviewVerdictFolder`.

The terminal :class:`~molexp.agent.harness.events.ModeCompletedEvent`
carries an :class:`~molexp.agent.mode.AgentRunResult` whose
``mode_state["verdict"]`` holds the JSON-mode :class:`ReviewVerdict`.
When the outcome is ``needs_changes`` the verdict's
:class:`~molexp.agent.modes._planning.PlanDiff` is surfaced as a
``repair_proposed`` event so it feeds the shared repair loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
)
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
)
from molexp.agent.modes.review.checks import (
    check_capability_evidence,
    check_intent_conformance,
    check_lifecycle_consistency,
)
from molexp.agent.modes.review.target import (
    ReviewTarget,
    ReviewTargetKind,
    detect_review_target,
)
from molexp.agent.modes.review.verdict import (
    ReviewVerdict,
    StepFinding,
    build_review_verdict,
)
from molexp.agent.modes.review.verdict_folder import ReviewVerdictFolder
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan.plan_folder import PlanFolder

_LOG = get_logger(__name__)

__all__ = ["ReviewMode", "ReviewModeConfig"]

_REVIEW_SUMMARY_SYSTEM = (
    "You summarize an agent's plan review in a single neutral sentence. Output only the sentence."
)


class ReviewModeConfig(BaseModel):
    """Tunables for :class:`ReviewMode`.

    Attributes:
        verdict_folder_name: Name of the
            :class:`~molexp.agent.modes.review.verdict_folder.ReviewVerdictFolder`
            ReviewMode mounts under its plan anchor.
        request_llm_summary: When ``True`` (the default) ReviewMode asks
            the router for a one-line natural-language summary; it
            degrades cleanly to a deterministic summary when the router
            cannot complete text.
    """

    model_config = ConfigDict(frozen=True)

    verdict_folder_name: str = "review"
    request_llm_summary: bool = True


class _ReviewOutcome:
    """Mutable scratchpad carrying the three stages' artefacts.

    Plain runtime container — the pipeline mutates it across stages.
    """

    def __init__(self) -> None:
        self.target: ReviewTarget | None = None
        self.findings: tuple[StepFinding, ...] = ()
        self.summary: str = ""
        self.verdict: ReviewVerdict | None = None


class ReviewMode(AgentMode):
    """Read-only typed review of an existing artefact — three stages."""

    name = "review"
    pipeline = ModePipeline(
        stages=(
            "IngestReviewTarget",
            "RunReviewChecks",
            "RenderReviewVerdict",
        ),
        edges=(
            PipelineEdge(from_stage="IngestReviewTarget", to_stage="RunReviewChecks"),
            PipelineEdge(from_stage="RunReviewChecks", to_stage="RenderReviewVerdict"),
            PipelineEdge(from_stage="RenderReviewVerdict", to_stage="verdict_rendered"),
        ),
        terminal_states=("verdict_rendered",),
    )

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
        """Construct a ReviewMode bound to one reviewable artefact.

        Args:
            config: Mode tunables; the default
                :class:`ReviewModeConfig` is used when omitted.
            plan_folder: The plan anchor — the verdict folder mounts
                under it.
            intent: The :class:`IntentSpec` the artefact is judged
                against.
            plan_graph: The typed plan under review (the ``plan`` kind),
                or the plan a workspace / run was materialized from.
            capability_graph: The :class:`CapabilityGraph` the plan was
                built against.
            workspace_path: A materialized-workspace path, when the
                target is a ``workspace``.
            run_ref: A completed-``Run`` id, when the target is a ``run``.
        """
        self.config = config or ReviewModeConfig()
        self.plan_folder = plan_folder
        self._intent = intent
        self._plan_graph = plan_graph
        self._capability_graph = capability_graph
        self._workspace_path = workspace_path
        self._run_ref = run_ref

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
        outcome = _ReviewOutcome()

        await self._ingest(harness, user_input, outcome)
        await self._run_checks(harness, outcome)
        await self._render(harness, outcome)

        yield self._completion(harness, outcome)

    # ── stage 1 — ingest ─────────────────────────────────────────────────

    async def _ingest(
        self, harness: AgentHarness, user_input: str, outcome: _ReviewOutcome
    ) -> None:
        """Detect the :class:`ReviewTarget` from the bound inputs."""
        async with harness.stage("IngestReviewTarget"):
            # `workspace` / `run` targets still review the typed plan they
            # were materialized from — the plan graph stays bound — but the
            # detected kind reflects what the user pointed at.
            target = self._detect_target(user_input)
            outcome.target = target
        _LOG.info(f"[review] ingested target kind={target.kind.value}")

    def _detect_target(self, user_input: str) -> ReviewTarget:
        """Resolve the :class:`ReviewTarget` for this review.

        An explicit ``workspace_path`` / ``run_ref`` selects that kind;
        otherwise the inline :class:`PlanGraph` is the target.
        """
        if self._workspace_path is not None:
            return detect_review_target(
                user_input=user_input,
                plan_graph=None,
                workspace_path=self._workspace_path,
                run_ref=None,
            )
        if self._run_ref is not None:
            return detect_review_target(
                user_input=user_input,
                plan_graph=None,
                workspace_path=None,
                run_ref=self._run_ref,
            )
        return detect_review_target(
            user_input=user_input,
            plan_graph=self._plan_graph,
            workspace_path=None,
            run_ref=None,
        )

    # ── stage 2 — checks ─────────────────────────────────────────────────

    async def _run_checks(self, harness: AgentHarness, outcome: _ReviewOutcome) -> None:
        """Run the three checkers and collect a one-line summary."""
        async with harness.stage("RunReviewChecks"):
            findings = (
                *check_intent_conformance(self._intent, self._plan_graph),
                *check_capability_evidence(self._plan_graph, self._capability_graph),
                *check_lifecycle_consistency(self._plan_graph),
            )
            outcome.findings = findings
            outcome.summary = await self._summarize(harness, findings)
        _LOG.info(f"[review] {len(outcome.findings)} finding(s) from three checkers")

    async def _summarize(self, harness: AgentHarness, findings: tuple[StepFinding, ...]) -> str:
        """Ask the router for a one-line summary; degrade cleanly.

        Returns an empty string when ``request_llm_summary`` is off or the
        router cannot complete text — :func:`build_review_verdict` then
        derives a deterministic summary.
        """
        if not self.config.request_llm_summary:
            return ""
        router = harness.router
        complete_text = getattr(router, "complete_text", None)
        if not callable(complete_text):
            return ""
        prompt = self._summary_prompt(findings)
        try:
            result = await complete_text(
                prompt=prompt,
                system=_REVIEW_SUMMARY_SYSTEM,
                tier=ModelTier.CHEAP,
            )
        except Exception as exc:
            _LOG.warning(f"[review] summary call failed, using deterministic fallback: {exc}")
            return ""
        return str(getattr(result, "text", "")).strip()

    @staticmethod
    def _summary_prompt(findings: tuple[StepFinding, ...]) -> str:
        """Render the findings into a prompt for the summary call."""
        if not findings:
            return "The review found no issues. Summarize in one sentence."
        rendered = "\n".join(f"- [{f.severity}] {f.summary}" for f in findings)
        return f"Review findings:\n{rendered}\n\nSummarize in one sentence."

    # ── stage 3 — render + persist ───────────────────────────────────────

    async def _render(self, harness: AgentHarness, outcome: _ReviewOutcome) -> None:
        """Fold the findings into a verdict, persist it, emit events."""
        async with harness.stage("RenderReviewVerdict"):
            assert outcome.target is not None
            verdict = build_review_verdict(
                findings=outcome.findings,
                intent=self._intent,
                plan=self._plan_graph,
                target_kind=outcome.target.kind,
                summary=outcome.summary,
            )
            outcome.verdict = verdict
            yaml_path, _ = self._persist(verdict)

        await harness.emit(ArtifactWrittenEvent(path=str(yaml_path), description="ReviewVerdict"))
        if verdict.overall == "needs_changes" and verdict.plan_diff is not None:
            await harness.emit(
                RepairProposedEvent(
                    failed_invariant=verdict.plan_diff.failed_invariant,
                    rationale=verdict.plan_diff.rationale,
                )
            )

    def _persist(self, verdict: ReviewVerdict) -> tuple[Path, Path]:
        """Persist the verdict through a plan-anchored :class:`ReviewVerdictFolder`.

        This is ReviewMode's *only* write surface — no ``src/`` /
        ``tests/`` / ``ir/`` directory is ever touched.
        """
        folder = self._verdict_folder()
        return folder.write_verdict(verdict)

    def _verdict_folder(self) -> ReviewVerdictFolder:
        """Mount (idempotently) the plan-anchored :class:`ReviewVerdictFolder`."""
        name = self.config.verdict_folder_name
        if not self.plan_folder.has_folder(name, cls=ReviewVerdictFolder):
            self.plan_folder.add_folder(
                ReviewVerdictFolder(name=name, plan_id=self.plan_folder.plan_id)
            )
        return self.plan_folder.get_folder(name, cls=ReviewVerdictFolder)

    # ── completion ───────────────────────────────────────────────────────

    def _completion(self, harness: AgentHarness, outcome: _ReviewOutcome) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        assert outcome.verdict is not None
        verdict = outcome.verdict
        breakdown = harness.router.snapshot_usage()
        kind = outcome.target.kind if outcome.target is not None else ReviewTargetKind.plan
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
