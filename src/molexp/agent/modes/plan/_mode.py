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

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes.plan._repair_loop import drive_with_repair
from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY, PlanModelPolicy
from molexp.agent.modes.plan.protocols import CapabilityProbe, PlanDeps
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    HandoffResult,
    PlanBriefResult,
    SkeletonResult,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
from molexp.agent.review import BypassPolicy, ReviewPolicy
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


__all__ = ["PlanMode", "PlanModeConfig", "PlanResult"]


_LOG = get_logger(__name__)


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
    - ``model_policy`` — :class:`PlanModelPolicy` defining
      tier-per-node assignments. Defaults to ``STANDARD_PLAN_POLICY``.
    - ``step_policy`` — initial per-step
      :class:`~molexp.agent.review.ReviewPolicy` fired after every
      non-terminal node's ``_execute``.  Defaults to
      :class:`~molexp.agent.review.BypassPolicy` (never blocks).
      Hot-swappable mid-run via :meth:`set_step_policy`.
    - ``final_policy`` — initial plan-final
      :class:`~molexp.agent.review.ReviewPolicy` consulted by
      ``HumanReview``.  Defaults to :class:`BypassPolicy` (auto-approve)
      so existing scripts that never wired up a gate keep their
      non-interactive behaviour.  Hot-swappable via
      :meth:`set_final_policy`.
    - ``artifacts_root``, ``max_iterations``, ``temperature`` — spare
      knobs threaded into :class:`PlanModeConfig`. ``max_iterations``
      caps the review→repair loop budget; the others are
      informational placeholders.

    The :class:`Router` is supplied by :class:`AgentRunner` at run
    time. PlanMode does not take a ``router=`` kwarg — model
    configuration is a runner-level concern. Tests that need a custom
    dispatcher inject one through ``AgentRunner(router=<fake>)`` and
    drive ``runner.run(...)``; tests that drive the workflow directly
    (``PLAN_WORKFLOW.execute(...)``) build :class:`PlanDeps`
    explicitly and supply the router on that side.

    Mid-run policy swap
    -------------------

    Use :meth:`set_step_policy` / :meth:`get_step_policy` (and the
    ``final_*`` pair) to replace or inspect the active policies at any
    time during a running plan — for example from a UI button, a
    SIGINT handler, or another coroutine that decides "stop asking,
    just go"::

        mode = PlanMode(workspace_handle=..., final_policy=HumanPolicy())
        # ... runner.run(...) executes in a task ...
        mode.set_final_policy(BypassPolicy())  # next HumanReview auto-approves

    The deps each node receives carries
    :attr:`~molexp.agent.modes.plan.protocols.PlanDeps.step_policy_lookup`
    and ``final_policy_lookup`` callables that close over this
    instance's ``_step_policy`` / ``_final_policy`` slots, so each
    consultation reads the latest assigned policy.
    """

    name = "plan"

    def __init__(
        self,
        *,
        workspace_handle: PlanWorkspaceHandle,
        model_policy: PlanModelPolicy | None = None,
        step_policy: ReviewPolicy | None = None,
        final_policy: ReviewPolicy | None = None,
        capability_probe: CapabilityProbe | None = None,
        artifacts_root: Path | None = None,
        max_iterations: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.config = PlanModeConfig(
            artifacts_root=artifacts_root,
            max_iterations=max_iterations,
            temperature=temperature,
        )
        self._workspace_handle = workspace_handle
        self._model_policy = model_policy if model_policy is not None else STANDARD_PLAN_POLICY
        self._step_policy: ReviewPolicy = step_policy if step_policy is not None else BypassPolicy()
        self._final_policy: ReviewPolicy = (
            final_policy if final_policy is not None else BypassPolicy()
        )
        self._capability_probe: CapabilityProbe | None = capability_probe

    def get_step_policy(self) -> ReviewPolicy:
        """Return the current per-step review policy."""
        return self._step_policy

    def set_step_policy(self, policy: ReviewPolicy) -> None:
        """Replace the active per-step policy; takes effect on the next node."""
        self._step_policy = policy

    def get_final_policy(self) -> ReviewPolicy:
        """Return the current plan-final review policy consulted by ``HumanReview``."""
        return self._final_policy

    def set_final_policy(self, policy: ReviewPolicy) -> None:
        """Replace the active plan-final policy; takes effect on the next review."""
        self._final_policy = policy

    def get_capability_probe(self) -> CapabilityProbe | None:
        """Return the configured :class:`CapabilityProbe`, or ``None``.

        ``None`` means PlanMode will resolve to a
        :class:`~molexp.agent.modes.plan.tasks_capability.NullCapabilityProbe`
        at workflow run time. :class:`AgentRunner` calls this method
        before constructing its lazy probe; a non-``None`` return
        means the user has already set one explicitly and the runner
        leaves it alone.
        """
        return self._capability_probe

    def set_capability_probe(self, probe: CapabilityProbe | None) -> None:
        """Inject a :class:`CapabilityProbe` for use by the discovery nodes.

        Called by :class:`AgentRunner` during ``run()`` to wire in a
        :class:`PydanticAICapabilityProbe` when molmcp is configured.
        Tests construct PlanMode with the probe directly via the
        constructor; this setter exists so the runner doesn't have to
        re-create the mode just to inject the probe.
        """
        self._capability_probe = probe

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        router.clear_usage()
        session.append(Message(role="user", content=user_input))
        _LOG.info(
            f"[plan] start plan_id={self._workspace_handle.plan_id} "
            f"workspace={self._workspace_handle.root()}"
        )
        _LOG.debug(
            f"[plan-mode] max_iterations={self.config.max_iterations} input_chars={len(user_input)}"
        )

        deps = PlanDeps(
            router=router,
            policy=self._model_policy,
            workspace_handle=self._workspace_handle,
            step_policy_lookup=lambda: self._step_policy,
            final_policy_lookup=lambda: self._final_policy,
            capability_probe=self._capability_probe,
        )
        t0 = time.monotonic()
        result = await drive_with_repair(
            deps, user_input, max_iterations=self.config.max_iterations
        )
        elapsed = time.monotonic() - t0
        breakdown = router.snapshot_usage()

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
                f"{self._workspace_handle.root()}."
            )
        elif isinstance(skeleton_result, SkeletonResult):
            summary = (
                f"PlanMode materialized {skeleton_result.workflow_py_path} "
                f"under {self._workspace_handle.root()}."
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
            "workspace_path": self._workspace_handle.root(),
            "outputs": {
                name: (payload.model_dump() if isinstance(payload, BaseModel) else payload)
                for name, payload in outputs.items()
            },
            "plan": plan_compat,
        }

        session.append(Message(role="assistant", content=view.summary))
        session.mode_state["plan"] = plan_compat
        _LOG.info(
            f"[plan] done in {elapsed:.1f}s: status={status} "
            f"approved={approved} ready_for_run={ready_for_run}"
        )
        _LOG.info(
            f"[plan] LLM usage: {breakdown.total.requests} request(s), "
            f"{breakdown.total.total_tokens} token(s)"
        )
        # Multi-line breakdown table at the end so it's the last thing
        # the user sees; one chunk so it doesn't get interleaved with
        # other loggers' output.
        _LOG.debug("[plan-mode] usage breakdown:\n" + breakdown.render_table())
        return AgentRunResult(
            text=view.summary,
            messages=tuple(session.history),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
