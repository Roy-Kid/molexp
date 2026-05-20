"""Review-target cluster — what ReviewMode is asked to review.

``ReviewMode`` reviews an *existing* artefact against the shared typed
contracts. Three artefact kinds are reviewable:

- :data:`ReviewTargetKind.plan` — a typed
  :class:`~molexp.agent.modes._planning.PlanGraph`, supplied inline.
- :data:`ReviewTargetKind.workspace` — a materialized experiment
  workspace (a ``src/`` tree AuthorMode produced), referenced by path.
- :data:`ReviewTargetKind.run` — a completed workspace ``Run``,
  referenced by id.

:class:`ReviewTarget` is the frozen typed reference; exactly one of its
three optional payload fields is populated per kind.
:func:`detect_review_target` resolves a kind from the inputs the mode
holds — an inline plan graph always wins, then a workspace path, then a
run reference.

Pure frozen-pydantic data + a pure function; no LLM, no I/O.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from molexp.agent.modes._planning import PlanGraph

__all__ = ["ReviewTarget", "ReviewTargetKind", "detect_review_target"]


class ReviewTargetKind(StrEnum):
    """The kind of artefact ReviewMode is asked to review."""

    plan = "plan"
    workspace = "workspace"
    run = "run"


class ReviewTarget(BaseModel):
    """A typed reference to the artefact under review.

    Exactly one payload field is populated, matching :attr:`kind`:
    ``plan`` → :attr:`plan_graph`, ``workspace`` → :attr:`workspace_path`,
    ``run`` → :attr:`run_ref`.

    Attributes:
        kind: Which artefact kind this target points at.
        plan_graph: The inline typed plan, when ``kind`` is ``plan``.
        workspace_path: The materialized-workspace path, when ``kind`` is
            ``workspace``.
        run_ref: The completed-``Run`` id, when ``kind`` is ``run``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    kind: ReviewTargetKind
    plan_graph: PlanGraph | None = None
    workspace_path: Path | None = None
    run_ref: str | None = None

    @model_validator(mode="after")
    def _check_payload_matches_kind(self) -> ReviewTarget:
        """Enforce that the populated payload field matches :attr:`kind`."""
        expected = {
            ReviewTargetKind.plan: self.plan_graph is not None,
            ReviewTargetKind.workspace: self.workspace_path is not None,
            ReviewTargetKind.run: self.run_ref is not None,
        }
        if not expected[self.kind]:
            raise ValueError(f"a {self.kind.value!r} ReviewTarget must carry its matching payload")
        return self


def detect_review_target(
    *,
    user_input: str,
    plan_graph: PlanGraph | None,
    workspace_path: Path | None,
    run_ref: str | None,
) -> ReviewTarget:
    """Resolve a :class:`ReviewTarget` from the review inputs.

    Detection order — an inline plan graph always wins (the typed plan is
    the richest target), then a materialized workspace path, then a
    completed-run reference. ``user_input`` is accepted for parity with
    the mode contract but the typed inputs are authoritative.

    Args:
        user_input: The free-text review request (informational).
        plan_graph: An inline typed plan, or ``None``.
        workspace_path: A materialized-workspace path, or ``None``.
        run_ref: A completed-``Run`` id, or ``None``.

    Returns:
        The resolved :class:`ReviewTarget`.

    Raises:
        ValueError: when no reviewable artefact is supplied.
    """
    _ = user_input
    if plan_graph is not None:
        return ReviewTarget(kind=ReviewTargetKind.plan, plan_graph=plan_graph)
    if workspace_path is not None:
        return ReviewTarget(kind=ReviewTargetKind.workspace, workspace_path=workspace_path)
    if run_ref is not None:
        return ReviewTarget(kind=ReviewTargetKind.run, run_ref=run_ref)
    raise ValueError("detect_review_target needs a plan_graph, workspace_path, or run_ref")
