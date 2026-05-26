"""ReviewMode's three Stage subclasses.

- :class:`IngestReviewTarget` — detect the :class:`ReviewTarget`
- :class:`RunReviewChecks` — run the three checkers + optional LLM summary
- :class:`RenderReviewVerdict` — fold into :class:`ReviewVerdict` and persist

Stages store their results on the bound :class:`ReviewMode` instance
(``_target`` / ``_findings`` / ``_summary`` / ``_verdict``); ReviewMode
reads them post-pipeline to build the terminal
:class:`ModeCompletedEvent`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.agent.harness.events import (
    AgentEvent,
    ArtifactWrittenEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.stage import Stage
from molexp.agent.modes.review.checks import (
    check_capability_evidence,
    check_intent_conformance,
    check_lifecycle_consistency,
)
from molexp.agent.modes.review.target import detect_review_target
from molexp.agent.modes.review.verdict import (
    ReviewVerdict,
    StepFinding,
    build_review_verdict,
)
from molexp.agent.modes.review.verdict_folder import ReviewVerdictFolder
from molexp.agent.router import ModelTier

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.review._mode import ReviewMode

__all__ = ["IngestReviewTarget", "RenderReviewVerdict", "RunReviewChecks"]

_LOG = get_logger(__name__)

_REVIEW_SUMMARY_SYSTEM = (
    "You summarize an agent's plan review in a single neutral sentence. Output only the sentence."
)


class IngestReviewTarget(Stage[str, str]):
    """Stage 1 — detect the :class:`ReviewTarget` from the bound inputs."""

    name: ClassVar[str] = "IngestReviewTarget"

    def __init__(self, *, review_mode: ReviewMode) -> None:
        self._mode = review_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: str,
    ) -> AsyncIterator[AgentEvent | str]:
        mode = self._mode
        if mode._workspace_path is not None:
            target = detect_review_target(
                user_input=input,
                plan_graph=None,
                workspace_path=mode._workspace_path,
                run_ref=None,
            )
        elif mode._run_ref is not None:
            target = detect_review_target(
                user_input=input,
                plan_graph=None,
                workspace_path=None,
                run_ref=mode._run_ref,
            )
        else:
            target = detect_review_target(
                user_input=input,
                plan_graph=mode._plan_graph,
                workspace_path=None,
                run_ref=None,
            )
        mode._target = target
        _LOG.info(f"[review] ingested target kind={target.kind.value}")
        yield input


class RunReviewChecks(Stage[str, str]):
    """Stage 2 — run the three checkers + optional LLM summary."""

    name: ClassVar[str] = "RunReviewChecks"

    def __init__(self, *, review_mode: ReviewMode) -> None:
        self._mode = review_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: str,
    ) -> AsyncIterator[AgentEvent | str]:
        mode = self._mode
        findings = (
            *check_intent_conformance(mode._intent, mode._plan_graph),
            *check_capability_evidence(mode._plan_graph),
            *check_lifecycle_consistency(mode._plan_graph),
        )
        mode._findings = findings
        mode._summary = await self._summarize(harness, findings)
        _LOG.info(f"[review] {len(mode._findings)} finding(s) from three checkers")
        yield input

    async def _summarize(self, harness: AgentHarness, findings: tuple[StepFinding, ...]) -> str:
        if not self._mode.config.request_llm_summary:
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
        except Exception as exc:  # pragma: no cover — degrade-cleanly path
            _LOG.warning(f"[review] summary call failed, using deterministic fallback: {exc}")
            return ""
        return str(getattr(result, "text", "")).strip()

    @staticmethod
    def _summary_prompt(findings: tuple[StepFinding, ...]) -> str:
        if not findings:
            return "The review found no issues. Summarize in one sentence."
        rendered = "\n".join(f"- [{f.severity}] {f.summary}" for f in findings)
        return f"Review findings:\n{rendered}\n\nSummarize in one sentence."


class RenderReviewVerdict(Stage[str, str]):
    """Stage 3 — fold findings into a :class:`ReviewVerdict` and persist."""

    name: ClassVar[str] = "RenderReviewVerdict"

    def __init__(self, *, review_mode: ReviewMode) -> None:
        self._mode = review_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: str,
    ) -> AsyncIterator[AgentEvent | str]:
        mode = self._mode
        assert mode._target is not None, "RenderReviewVerdict ran before IngestReviewTarget"
        verdict = build_review_verdict(
            findings=mode._findings,
            intent=mode._intent,
            plan=mode._plan_graph,
            target_kind=mode._target.kind,
            summary=mode._summary,
        )
        mode._verdict = verdict
        yaml_path, _ = self._persist(verdict)

        # Review's events go through ``harness.emit`` (sink-only); they
        # don't trigger any RepairPolicy on the pipeline so the executor
        # doesn't need to observe them via yield. The stage's terminal
        # yield is the input pass-through.
        await harness.emit(ArtifactWrittenEvent(path=str(yaml_path), description="ReviewVerdict"))
        if verdict.overall == "needs_changes" and verdict.plan_diff is not None:
            await harness.emit(
                RepairProposedEvent(
                    failed_invariant=verdict.plan_diff.failed_invariant,
                    rationale=verdict.plan_diff.rationale,
                )
            )
        yield input

    def _persist(self, verdict: ReviewVerdict) -> tuple[Path, Path]:
        folder = self._verdict_folder()
        return folder.write_verdict(verdict)

    def _verdict_folder(self) -> ReviewVerdictFolder:
        mode = self._mode
        name = mode.config.verdict_folder_name
        if not mode.plan_folder.has_folder(name, cls=ReviewVerdictFolder):
            mode.plan_folder.add_folder(
                ReviewVerdictFolder(name=name, plan_id=mode.plan_folder.plan_id)
            )
        return mode.plan_folder.get_folder(name, cls=ReviewVerdictFolder)
