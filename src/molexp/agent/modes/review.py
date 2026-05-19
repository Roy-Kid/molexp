"""``ReviewMode`` — phase-2 workflow skeleton.

Three-task placeholder workflow:

    IngestReviewTarget → SummarizeReviewTarget → RenderReviewVerdict

Each task raises :class:`NotImplementedError` until phase 2 fills in
the bodies; the workflow plumbing (wiring, deps threading) is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.workflow import Task, TaskContext, Workflow, WorkflowBuilder

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


_LOG = get_logger(__name__)


class ReviewModeConfig(BaseModel):
    """Tunables for :class:`ReviewMode`.

    Attributes:
        review_target: Identifier surfaced into the workflow's
            ``config["review_target"]``. Phase 2 will narrow this to a
            typed reference.
    """

    model_config = ConfigDict(frozen=True)

    review_target: str = ""


@dataclass(frozen=True)
class _ReviewDeps:
    """Runtime services for the review workflow."""

    router: Router


class IngestReviewTarget(Task):
    """Phase-2 placeholder — loads the target artefact."""

    async def execute(self, ctx: TaskContext[None, _ReviewDeps, None]) -> None:
        del ctx
        raise NotImplementedError("ReviewMode.IngestReviewTarget is reserved for phase 2")


class SummarizeReviewTarget(Task):
    """Phase-2 placeholder — summarises the target via the router."""

    async def execute(self, ctx: TaskContext[None, _ReviewDeps, None]) -> None:
        del ctx
        raise NotImplementedError("ReviewMode.SummarizeReviewTarget is reserved for phase 2")


class RenderReviewVerdict(Task):
    """Phase-2 placeholder — produces the structured verdict."""

    async def execute(self, ctx: TaskContext[None, _ReviewDeps, None]) -> None:
        del ctx
        raise NotImplementedError("ReviewMode.RenderReviewVerdict is reserved for phase 2")


def build_review_workflow() -> Workflow:
    """Assemble the three-task review skeleton."""
    builder = WorkflowBuilder(name="review_mode", entry="IngestReviewTarget")
    builder.add(IngestReviewTarget(), name="IngestReviewTarget")
    builder.add(
        SummarizeReviewTarget(),
        name="SummarizeReviewTarget",
        depends_on=["IngestReviewTarget"],
    )
    builder.add(
        RenderReviewVerdict(),
        name="RenderReviewVerdict",
        depends_on=["SummarizeReviewTarget"],
    )
    return builder.build()


REVIEW_WORKFLOW: Workflow = build_review_workflow()
"""Module-level frozen review workflow skeleton."""


class ReviewMode(AgentMode):
    """Workflow-native review mode. Phase 2 fills in the task bodies."""

    name = "review"

    def __init__(self, *, config: ReviewModeConfig | None = None) -> None:
        self.config = config or ReviewModeConfig()

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        raise NotImplementedError(
            "ReviewMode is reserved for phase 2; the placeholder tasks "
            "always raise NotImplementedError today."
        )


__all__ = [
    "REVIEW_WORKFLOW",
    "IngestReviewTarget",
    "RenderReviewVerdict",
    "ReviewMode",
    "ReviewModeConfig",
    "SummarizeReviewTarget",
    "build_review_workflow",
]
