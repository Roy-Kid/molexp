"""Concrete tasks for the PlanMode workflow.

Each LLM-bearing task subclasses :class:`PlanLLMTask`, declares its
:class:`ModelTier`, and returns a structured pydantic schema. Pure
control / policy tasks subclass :class:`PlanTask` directly. No task
constructs an LLM client itself; every model call goes through
``ctx.deps.provider``.

There are no aggregator / compose tasks — every workflow node does
work that can stand on its own (LLM call, deterministic check, branch
decision, structural transform). Where multiple consumers need a
:class:`PlanSpec` view of the same upstream specs, each consumer
materialises one inline via :func:`compose_plan_spec`; the helper is a
function, not a workflow stage.

``ctx.inputs`` shape (molexp.workflow API): ``None`` for zero-dep
tasks, the bare upstream value for single-dep tasks, and a
``dict[name → value]`` for ≥ 2 deps. Each task asserts the shape it
expects in-line so the cast is explicit at the read site.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from molexp.agent.modes.plan.protocols import ModelTier, PlanDeps
from molexp.agent.modes.plan.schemas import (
    ApprovedPlan,
    CodegenOutput,
    CompileReport,
    ContextSpec,
    Decomposition,
    DryRunReport,
    ExecutableWorkflowDraft,
    GoalSpec,
    IntakeSpec,
    MethodSpec,
    PlanPreview,
    PlanSpec,
    ProtocolDraft,
    RepairReport,
)
from molexp.workflow import Task, TaskContext
from molexp.workflow.types import Next

# ── Helpers (functions, not tasks) ─────────────────────────────────────────


def compose_plan_spec(
    inputs: Mapping[str, Any],
    *,
    revision: int,
) -> PlanSpec:
    """Build a frozen :class:`PlanSpec` view from the upstream specs.

    Used by :class:`PreviewTask` and :class:`CodegenTask`; both have
    identical data deps, so each calls this directly rather than
    routing through a third aggregator workflow node.
    """
    return PlanSpec(
        goal=inputs["goal"],
        context=inputs["context"],
        method=inputs["method"],
        decomposition=inputs["decomposition"],
        protocol=inputs["protocol"],
        revision=revision,
    )


# ── Bases ──────────────────────────────────────────────────────────────────


class PlanTask(Task):
    """Base for every task inside the plan-mode workflow."""


class PlanLLMTask(PlanTask):
    """Plan task that invokes the provider to produce a structured output."""

    TIER: ClassVar[ModelTier] = ModelTier.DEFAULT
    SYSTEM_PROMPT: ClassVar[str] = ""

    async def invoke_llm[SchemaT](
        self,
        ctx: TaskContext[None, PlanDeps, Any],
        *,
        user: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        return await ctx.deps.provider.invoke(
            tier=self.TIER,
            system=self.SYSTEM_PROMPT,
            user=user,
            schema=schema,
            node_id=type(self).__name__,
        )


# ── Specification stages ───────────────────────────────────────────────────


class IntakeTask(PlanLLMTask):
    TIER = ModelTier.CHEAP
    SYSTEM_PROMPT = "Extract a structured IntakeSpec from the user request."

    async def execute(self, ctx: TaskContext[None, PlanDeps, None]) -> IntakeSpec:
        request: str = ctx.config["user_input"]  # type: ignore[assignment]
        return await self.invoke_llm(ctx, user=request, schema=IntakeSpec)


class GoalTask(PlanLLMTask):
    TIER = ModelTier.CHEAP
    SYSTEM_PROMPT = "Restate the user's goal as a precise GoalSpec."

    async def execute(self, ctx: TaskContext[None, PlanDeps, IntakeSpec]) -> GoalSpec:
        intake = ctx.inputs
        return await self.invoke_llm(ctx, user=intake.model_dump_json(), schema=GoalSpec)


class ContextTask(PlanLLMTask):
    TIER = ModelTier.CHEAP
    SYSTEM_PROMPT = "Identify constraints, assumptions and environment for this goal."

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, GoalSpec]
    ) -> ContextSpec:
        goal = ctx.inputs
        return await self.invoke_llm(ctx, user=goal.model_dump_json(), schema=ContextSpec)


class MethodTask(PlanLLMTask):
    TIER = ModelTier.DEFAULT
    SYSTEM_PROMPT = "Choose a concrete method that satisfies the goal under the context."

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> MethodSpec:
        goal: GoalSpec = ctx.inputs["goal"]
        context: ContextSpec = ctx.inputs["context"]
        user = (
            f"Goal: {goal.model_dump_json()}\n"
            f"Context: {context.model_dump_json()}"
        )
        return await self.invoke_llm(ctx, user=user, schema=MethodSpec)


class DecompositionTask(PlanLLMTask):
    TIER = ModelTier.HEAVY
    SYSTEM_PROMPT = "Break this method into ordered protocol stages."

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, MethodSpec]
    ) -> Decomposition:
        method = ctx.inputs
        return await self.invoke_llm(ctx, user=method.model_dump_json(), schema=Decomposition)


class ProtocolTask(PlanLLMTask):
    TIER = ModelTier.HEAVY
    SYSTEM_PROMPT = "Render each stage as a concrete protocol step."

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, Decomposition]
    ) -> ProtocolDraft:
        decomp = ctx.inputs
        return await self.invoke_llm(ctx, user=decomp.model_dump_json(), schema=ProtocolDraft)


# ── Preview ────────────────────────────────────────────────────────────────


class PreviewTask(PlanTask):
    """Render a frozen :class:`PlanPreview` from the upstream specs."""

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> PlanPreview:
        plan = compose_plan_spec(ctx.inputs, revision=ctx.deps.store.get_iteration())
        rendered = (
            f"Goal: {plan.goal.objective}\n"
            f"Method: {plan.method.name}\n"
            f"Stages: {', '.join(plan.decomposition.stages)}\n"
            f"Steps: {len(plan.protocol.steps)}\n"
            f"Revision: {plan.revision}"
        )
        return PlanPreview(plan=plan, rendered=rendered)


# ── Gate A ─────────────────────────────────────────────────────────────────


class GateATask(PlanTask):
    """Approve PlanSpec? — reads only the explicit ``preview`` input."""

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, PlanPreview]
    ) -> Next:
        preview = ctx.inputs
        decision = await ctx.deps.gate_policy.gate_a(preview)
        return Next("approve" if decision.approved else "patch")


# ── Codegen → executable draft ─────────────────────────────────────────────


class CodegenTask(PlanLLMTask):
    """Author one Python skeleton task per protocol stage and emit a draft.

    Reads the same upstream specs Preview does, calls the LLM for the
    generated code, and returns the executable draft directly — no
    separate ``ComposeExecutableWorkflowTask`` aggregator. Each
    :class:`GeneratedTaskSpec` is independently persisted via
    ``ctx.deps.artifact_writer`` so the generated workflow's nodes are
    addressable artefacts, not lines in a single blob.
    """

    TIER = ModelTier.HEAVY
    SYSTEM_PROMPT = (
        "For each protocol stage, author a Python Task subclass implementing "
        "it. Return one GeneratedTaskSpec per stage."
    )

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> ExecutableWorkflowDraft:
        plan = compose_plan_spec(ctx.inputs, revision=ctx.deps.store.get_iteration())
        codegen = await self.invoke_llm(
            ctx, user=plan.model_dump_json(), schema=CodegenOutput
        )
        for spec in codegen.generated:
            ctx.deps.artifact_writer.write(f"generated_task/{spec.task_id}", spec)
        return ExecutableWorkflowDraft(
            plan=plan,
            bound={gen.stage: gen.task_id for gen in codegen.generated},
            generated=codegen.generated,
        )


# ── Compile + Dry-run ──────────────────────────────────────────────────────


class CompileTask(PlanTask):
    """Compile the executable draft into a workflow template.

    Returns ``(report, Next)`` so the report lands on ``state.results``
    while the branch label dispatches the next step.
    """

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, ExecutableWorkflowDraft]
    ) -> tuple[CompileReport, Next]:
        executable = ctx.inputs
        ok = bool(executable.generated) or bool(executable.bound)
        report = CompileReport(
            ok=ok,
            workflow_template_id=f"tpl-{executable.plan.revision}" if ok else None,
            experiment_spec_id=f"exp-{executable.plan.revision}" if ok else None,
            diagnostics=() if ok else ("no bound or generated tasks",),
        )
        return report, Next("ok" if ok else "fail")


class DryRunTask(PlanTask):
    async def execute(
        self, ctx: TaskContext[None, PlanDeps, CompileReport]
    ) -> tuple[DryRunReport, Next]:
        report_in = ctx.inputs
        ok = report_in.ok
        report = DryRunReport(ok=ok, notes=() if ok else ("compile failed",))
        return report, Next("ok" if ok else "fail")


# ── Gate B ─────────────────────────────────────────────────────────────────


class GateBTask(PlanTask):
    """Approve handoff? — reads ``codegen`` + reports from explicit inputs."""

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> Next:
        executable: ExecutableWorkflowDraft = ctx.inputs["codegen"]
        compile_report: CompileReport = ctx.inputs["compile"]
        dry_run_report: DryRunReport = ctx.inputs["dry_run"]
        decision = await ctx.deps.gate_policy.gate_b(
            executable, compile_report, dry_run_report
        )
        return Next("approve" if decision.approved else "patch")


# ── Repair ─────────────────────────────────────────────────────────────────


class RepairTask(PlanTask):
    """Apply a :class:`RepairPolicy` patch and bump the iteration counter."""

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, PlanPreview]
    ) -> RepairReport:
        preview = ctx.inputs
        ctx.deps.store.note_iteration()
        report = await ctx.deps.repair_policy.patch(preview, reason="rejected")
        return RepairReport(
            iteration=ctx.deps.store.get_iteration(),
            patches=report.patches,
            affected_nodes=report.affected_nodes,
            stale_nodes=report.stale_nodes,
        )


# ── Handoff ────────────────────────────────────────────────────────────────


class HandoffTask(PlanTask):
    """Materialise the :class:`ApprovedPlan` payload for the runner."""

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> ApprovedPlan:
        executable: ExecutableWorkflowDraft = ctx.inputs["codegen"]
        compile_report: CompileReport = ctx.inputs["compile"]
        dry_run_report: DryRunReport = ctx.inputs["dry_run"]
        return ApprovedPlan(
            plan=executable.plan,
            executable=executable,
            compile_report=compile_report,
            dry_run_report=dry_run_report,
            iterations=ctx.deps.store.get_iteration(),
        )


__all__ = [
    "CodegenTask",
    "CompileTask",
    "ContextTask",
    "DecompositionTask",
    "DryRunTask",
    "GateATask",
    "GateBTask",
    "GoalTask",
    "HandoffTask",
    "IntakeTask",
    "MethodTask",
    "PlanLLMTask",
    "PlanTask",
    "PreviewTask",
    "ProtocolTask",
    "RepairTask",
    "compose_plan_spec",
]
