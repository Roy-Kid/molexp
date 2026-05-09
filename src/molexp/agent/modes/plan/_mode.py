"""``PlanMode`` + workflow assembly.

The workflow is a *single layer* of ``molexp.workflow`` tasks — no
aggregator / compose nodes. Multiple consumers that need a
:class:`~molexp.agent.modes.plan.schemas.PlanSpec` view materialise one
inline via :func:`~molexp.agent.modes.plan.tasks.compose_plan_spec`.

The cycle ``repair → preview`` is expressed as a plain ``wf.control``
back-edge; molexp.workflow accepts cycles in the control graph. There
is no runtime ``max_iters`` cap — gate policies are responsible for
termination.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY, PlanModelPolicy
from molexp.agent.modes.plan.protocols import (
    ArtifactWriter,
    AutoApproveGatePolicy,
    GatePolicy,
    IdentityRepairPolicy,
    InMemoryPlanStore,
    NoOpArtifactWriter,
    PlanDeps,
    PlanStore,
    Provider,
    RepairPolicy,
)
from molexp.agent.modes.plan.schemas import ApprovedPlan, PlanSpec
from molexp.agent.modes.plan.tasks import (
    CodegenTask,
    CompileTask,
    ContextTask,
    DecompositionTask,
    DryRunTask,
    GateATask,
    GateBTask,
    GoalTask,
    HandoffTask,
    IntakeTask,
    MethodTask,
    PreviewTask,
    ProtocolTask,
    RepairTask,
)
from molexp.agent.types import Message
from molexp.workflow import Workflow, WorkflowBuilder

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


# ── Public configs / results ────────────────────────────────────────────────


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`."""

    model_config = ConfigDict(frozen=True)

    artifacts_root: Path | None = None
    max_iterations: int = 8
    temperature: float | None = None


class PlanResult(BaseModel):
    """Frozen public summary of one :meth:`PlanMode.run` call.

    The full structured payload (``ApprovedPlan`` or, on repair-loop
    abort, the latest ``PlanPreview``) lives under
    ``AgentRunResult.mode_state``; this model exposes the minimal
    string-shaped pair that downstream callers / tests have always
    relied on plus an ``approved`` flag.
    """

    model_config = ConfigDict(frozen=True)

    intake: str
    design: str
    summary: str = ""
    approved: bool = False


# ── Workflow assembly (module-level) ───────────────────────────────────────


_PLAN_SPEC_DEPS = ["goal", "context", "method", "decomposition", "protocol"]


def _build_plan_workflow() -> Workflow:
    """Assemble the plan-mode workflow.

    Every task names its data deps; every branch names its routes; the
    ``repair → preview`` back-edge is a plain ``wf.control``. No
    aggregator tasks — Preview / Codegen each compose their own
    :class:`PlanSpec` view from the upstream specs via
    :func:`compose_plan_spec`.
    """

    wf = WorkflowBuilder(name="plan_mode", entry="intake")

    # Specification chain.
    wf.add(IntakeTask(), name="intake", next_="goal")
    wf.add(GoalTask(), name="goal", depends_on=["intake"], next_="context")
    wf.add(ContextTask(), name="context", depends_on=["goal"], next_="method")
    wf.add(
        MethodTask(),
        name="method",
        depends_on=["goal", "context"],
        next_="decomposition",
    )
    wf.add(
        DecompositionTask(),
        name="decomposition",
        depends_on=["method"],
        next_="protocol",
    )
    wf.add(
        ProtocolTask(),
        name="protocol",
        depends_on=["decomposition"],
        next_="preview",
    )

    # Preview reads every spec part directly — no compose_plan task.
    wf.add(
        PreviewTask(),
        name="preview",
        depends_on=list(_PLAN_SPEC_DEPS),
        next_="gate_a",
    )

    # Gate A → Codegen branch / Repair branch.
    wf.add(
        GateATask(),
        name="gate_a",
        depends_on=["preview"],
        routes={"approve": "codegen", "patch": "repair"},
    )

    # Codegen pulls the same upstream specs Preview did, runs the LLM,
    # and emits the executable draft directly.
    wf.add(
        CodegenTask(),
        name="codegen",
        depends_on=list(_PLAN_SPEC_DEPS),
        next_="compile",
    )
    wf.add(
        CompileTask(),
        name="compile",
        depends_on=["codegen"],
        routes={"ok": "dry_run", "fail": "repair"},
    )
    wf.add(
        DryRunTask(),
        name="dry_run",
        depends_on=["compile"],
        routes={"ok": "gate_b", "fail": "repair"},
    )
    wf.add(
        GateBTask(),
        name="gate_b",
        depends_on=["codegen", "compile", "dry_run"],
        routes={"approve": "handoff", "patch": "repair"},
    )

    # Repair loop — back-edge to preview. No max_iters cap; gate policies
    # own termination.
    wf.add(
        RepairTask(),
        name="repair",
        depends_on=["preview"],
        next_="preview",
    )

    # Terminal handoff.
    wf.add(
        HandoffTask(),
        name="handoff",
        depends_on=["codegen", "compile", "dry_run"],
    )

    return wf.build()


PLAN_WORKFLOW: Workflow = _build_plan_workflow()


# ── PlanMode ────────────────────────────────────────────────────────────────


class PlanMode(AgentMode):
    """Multi-step planning mode driven by a ``molexp.workflow`` spec.

    Construction takes only runtime services + tunables; the workflow
    itself is a module-level constant. All LLM calls flow through the
    injected :class:`Provider`; the ``harness`` parameter supplied by
    :class:`AgentRunner` to :meth:`run` is intentionally unused — model
    selection is a per-task tier declaration, not a runner-level choice.
    """

    name = "plan"

    def __init__(
        self,
        *,
        provider: Provider | None = None,
        gate_policy: GatePolicy | None = None,
        repair_policy: RepairPolicy | None = None,
        store: PlanStore | None = None,
        artifact_writer: ArtifactWriter | None = None,
        model_policy: PlanModelPolicy | None = None,
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
            gate_policy=gate_policy or AutoApproveGatePolicy(),
            repair_policy=repair_policy or IdentityRepairPolicy(),
            store=store or InMemoryPlanStore(),
            artifact_writer=artifact_writer or NoOpArtifactWriter(),
            model_policy=model_policy if model_policy is not None else STANDARD_PLAN_POLICY,
        )

    async def run(
        self,
        *,
        harness: PydanticAIHarness,  # noqa: ARG002 — runner-supplied; ignored by design (per-task tier policy)
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        session.append(Message(role="user", content=user_input))

        result = await PLAN_WORKFLOW.execute(
            config={"user_input": user_input},
            deps=self._deps,
        )

        outputs = result.outputs
        approved: ApprovedPlan | None = outputs.get("handoff")
        plan_spec: PlanSpec | None = (
            approved.plan
            if approved is not None
            else (outputs["preview"].plan if "preview" in outputs else None)
        )

        intake_obj = outputs.get("intake")
        intake_text = intake_obj.extracted_goal if intake_obj is not None else ""
        design_text = ""
        if plan_spec is not None:
            design_text = f"{plan_spec.method.name}\n\n" + "\n".join(
                f"- {step.stage}: {step.operation}" for step in plan_spec.protocol.steps
            )

        summary = (
            f"Plan approved (iterations={approved.iterations})."
            if approved is not None
            else "Plan not approved — handoff did not run."
        )
        view = PlanResult(
            intake=intake_text or "",
            design=design_text or "",
            summary=summary,
            approved=approved is not None,
        )

        mode_state: dict[str, Any] = {
            "approved_plan": approved.model_dump() if approved is not None else None,
            "plan_spec": plan_spec.model_dump() if plan_spec is not None else None,
            "outputs": {
                name: payload.model_dump() if isinstance(payload, BaseModel) else payload
                for name, payload in outputs.items()
            },
            # Back-compat shim for callers that read mode_state["plan"].
            "plan": {
                "intake": intake_text,
                "design": design_text,
                "approved": approved is not None,
                "iterations": approved.iterations if approved is not None else None,
            },
        }
        session.append(Message(role="assistant", content=view.summary))
        session.mode_state["plan"] = mode_state["plan"]
        return AgentRunResult(
            text=view.summary,
            messages=tuple(session.history),
            mode_state=mode_state,
        )


__all__ = ["PLAN_WORKFLOW", "PlanMode", "PlanModeConfig", "PlanResult"]
