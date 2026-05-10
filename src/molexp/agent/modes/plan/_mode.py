"""``PlanMode`` — the AgentMode adapter for the materialize-to-workspace pipeline.

The pipeline itself is a module-level :class:`~molexp.workflow.Workflow`
constant in :mod:`molexp.agent.modes.plan._pipeline`. This module
adapts :class:`AgentMode.run` to it: takes the user's input, executes
:data:`PLAN_WORKFLOW`, and projects the terminal
:class:`~molexp.workflow.WorkflowResult` into an
:class:`AgentRunResult` whose ``mode_state`` exposes the materialized
workspace path plus a back-compat shim under ``mode_state["plan"]``
for consumers that have not yet migrated to the v1 surface.

v1 (this sub-spec) does not run a human-review / approval node;
sub-spec 06 will. ``mode_state["plan"]["approved"]`` is therefore
``False`` and ``["iterations"]`` is ``None`` — this is the binding
contract per acceptance criterion ``ac-010``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes.plan._repair_loop import drive_with_repair
from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY, PlanModelPolicy
from molexp.agent.modes.plan.protocols import (
    AutoApproveGatePolicy,
    GatePolicy,
    PlanDeps,
    Provider,
)
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    HandoffResult,
    PlanBriefResult,
    SkeletonResult,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


__all__ = ["PlanMode", "PlanModeConfig", "PlanResult"]


# ── Public configs / results ────────────────────────────────────────────────


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`.

    Attributes:
        artifacts_root: Optional override for the workspace root
            directory; ``None`` means inherit from the supplied
            :class:`PlanWorkspaceHandle`.
        max_iterations: Reserved for sub-spec 06's repair loop.
            Currently unused — v1 has no repair cycle.
        temperature: Reserved for future provider tuning.
    """

    model_config = ConfigDict(frozen=True)

    artifacts_root: Path | None = None
    max_iterations: int = 8
    temperature: float | None = None


class PlanResult(BaseModel):
    """Frozen public summary of one :meth:`PlanMode.run` call.

    The full per-node payload (the six ``*Result`` instances) lives
    under :attr:`AgentRunResult.mode_state["outputs"]`; this view
    surfaces the back-compat ``intake`` / ``design`` strings plus the
    ``approved`` flag.
    """

    model_config = ConfigDict(frozen=True)

    intake: str
    design: str
    summary: str = ""
    approved: bool = False
    ready_for_run: bool = False
    status: str = ""


# ── PlanMode ────────────────────────────────────────────────────────────────


class PlanMode(AgentMode):
    """Materialize-to-workspace planning mode.

    Construction takes:

    - ``workspace_handle`` (required) — the
      :class:`PlanWorkspaceHandle` rooted at the experiment workspace
      this run will write into.
    - ``provider`` — LLM dispatch gateway. Defaults to a fresh
      ``PydanticAIProvider``; supply a stub for tests.
    - ``model_policy`` — :class:`PlanModelPolicy` defining
      tier-per-node assignments. Defaults to ``STANDARD_PLAN_POLICY``.
    - ``artifacts_root``, ``max_iterations``, ``temperature`` — spare
      knobs threaded into :class:`PlanModeConfig`. Currently
      informational only; sub-spec 06's repair loop will start using
      them.

    The ``harness`` parameter handed in by :class:`AgentRunner.run` is
    intentionally unused: model selection is a per-task tier policy,
    not a runner-level choice.
    """

    name = "plan"

    def __init__(
        self,
        *,
        workspace_handle: PlanWorkspaceHandle,
        provider: Provider | None = None,
        model_policy: PlanModelPolicy | None = None,
        gate_policy: GatePolicy | None = None,
        artifacts_root: Path | None = None,
        max_iterations: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.config = PlanModeConfig(
            artifacts_root=artifacts_root,
            max_iterations=max_iterations,
            temperature=temperature,
        )
        if provider is None:
            from molexp.agent._pydanticai.provider import PydanticAIProvider

            provider = PydanticAIProvider()
        self._deps = PlanDeps(
            provider=provider,
            policy=model_policy if model_policy is not None else STANDARD_PLAN_POLICY,
            workspace_handle=workspace_handle,
            gate_policy=gate_policy if gate_policy is not None else AutoApproveGatePolicy(),
        )

    async def run(
        self,
        *,
        harness: PydanticAIHarness,  # noqa: ARG002 — runner-supplied; ignored by design (per-task tier policy)
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        session.append(Message(role="user", content=user_input))

        result = await drive_with_repair(
            self._deps, user_input, max_iterations=self.config.max_iterations
        )

        outputs = result.outputs
        digest_result = outputs.get("DraftReportDigest")
        plan_brief_result = outputs.get("DraftImplementationPlan")
        skeleton_result = outputs.get("GenerateWorkflowSkeleton")
        handoff_result = outputs.get("FinalHandoffCheck")
        if not isinstance(handoff_result, HandoffResult):
            handoff_result = outputs.get("HumanReview")

        intake_text: str = ""
        design_text: str = ""
        if isinstance(digest_result, DigestResult):
            intake_text = digest_result.digest.experimental_goal or digest_result.digest.summary
        if isinstance(plan_brief_result, PlanBriefResult):
            brief = plan_brief_result.plan_brief
            design_text = brief.overview
            if brief.stages:
                design_text = f"{brief.overview}\n\n" + "\n".join(
                    f"- {stage}" for stage in brief.stages
                )

        if isinstance(handoff_result, HandoffResult):
            summary = (
                f"PlanMode {handoff_result.status} "
                f"plan_id={handoff_result.handoff.plan_id} "
                f"ready_for_run={handoff_result.ready_for_run} at "
                f"{self._deps.workspace_handle.root()}."
            )
        elif isinstance(skeleton_result, SkeletonResult):
            summary = (
                f"PlanMode materialized {skeleton_result.workflow_py_path} "
                f"under {self._deps.workspace_handle.root()}."
            )
        else:
            summary = "PlanMode pipeline did not reach the skeleton-generation step."

        approved = bool(
            isinstance(handoff_result, HandoffResult) and handoff_result.decision.approved
        )
        ready_for_run = bool(
            isinstance(handoff_result, HandoffResult) and handoff_result.ready_for_run
        )
        status = handoff_result.status if isinstance(handoff_result, HandoffResult) else "failed"
        view = PlanResult(
            intake=intake_text,
            design=design_text,
            summary=summary,
            approved=approved,
            ready_for_run=ready_for_run,
            status=status,
        )

        # Back-compat shim — preserves the ``mode_state["plan"]``
        # shape today's UI / tests rely on. Adds ``handoff`` (sub-spec
        # 06) when the gate approved; otherwise None.
        plan_compat: dict[str, Any] = {
            "intake": intake_text,
            "design": design_text,
            "approved": approved,
            "ready_for_run": ready_for_run,
            "status": status,
            "iterations": None,
            "handoff": (
                handoff_result.handoff.model_dump(mode="json")
                if isinstance(handoff_result, HandoffResult)
                else None
            ),
        }

        mode_state: dict[str, Any] = {
            "workspace_path": self._deps.workspace_handle.root(),
            "outputs": {
                name: (payload.model_dump() if isinstance(payload, BaseModel) else payload)
                for name, payload in outputs.items()
            },
            "plan": plan_compat,
        }

        session.append(Message(role="assistant", content=view.summary))
        session.mode_state["plan"] = plan_compat
        return AgentRunResult(
            text=view.summary,
            messages=tuple(session.history),
            mode_state=mode_state,
        )
