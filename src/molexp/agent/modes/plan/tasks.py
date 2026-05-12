"""Concrete tasks for the materialize-to-workspace PlanMode pipeline.

The 13 nodes — capability discovery runs *before* IR compilation so
the workflow IR / per-task IR are typed from real evidence instead of
guesses:

    IngestReport → DraftReportDigest → DraftImplementationPlan
        → DraftCapabilityNeeds → DiscoverCapabilities
        → CompileWorkflowIR → CompileTaskIR
        → GenerateWorkflowSkeleton
        → GenerateTaskTests / GenerateTaskImplementations
        → ValidateWorkspace → HumanReview → FinalHandoffCheck

Each node consumes its upstream ``ctx.inputs`` — a single bare value
when the node has one upstream, a ``dict[str, *Result]`` keyed by
upstream node name when it has more — materializes its product
through :class:`~molexp.agent.modes.plan.plan_folder.PlanFolder`,
and returns a frozen ``*Result`` carrying *path references* — never
embedded blobs. Downstream nodes operate on file handles, the
workspace is the single source of truth for materialized content.

LLM-bearing tasks subclass :class:`PlanLLMTask`; the tier each one
runs at is resolved by ``ctx.deps.policy.tier_for(type(self).__name__)``
(no per-class ``TIER`` ClassVar literal). ``IngestReport`` skips the
LLM entirely — it just hashes and writes the raw user input.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import importlib
import json
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, cast

import yaml
from mollog import get_logger
from pydantic import BaseModel

from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    validate_codegen_evidence,
)
from molexp.agent.modes.plan.context import (
    PLAN_PIPELINE_ORDER,
    format_node_label,
    format_progress_done,
    format_progress_start,
)
from molexp.agent.modes.plan.errors import (
    SkeletonCompileError,
    StepRejected,
    UnevidencedApiReference,
)
from molexp.agent.modes.plan.handoff import PlanRunHandoff
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    HandoffResult,
    IngestReportResult,
    PlanBrief,
    PlanBriefResult,
    PlanReviewView,
    ReportDigest,
    ReviewDecision,
    SkeletonResult,
    TaskImplementationModule,
    TaskImplementationsResult,
    TaskIRBrief,
    TaskIRResult,
    TaskTestModule,
    TaskTestsResult,
    ValidationResult,
    WorkflowContract,
    WorkflowIRResult,
)
from molexp.agent.modes.plan.plan_folder import (
    CheckResult,
    PlanFolder,
    PlanManifest,
    ValidationReport,
)
from molexp.agent.review import StepView
from molexp.workflow import (
    Task,
    TaskContext,
    TaskIO,
    Workflow,
    default_compiler,
    validate_workflow_contract,
)
from molexp.workspace import atomic_write_text

__all__ = [
    "CompileTaskIR",
    "CompileWorkflowIR",
    "DraftImplementationPlan",
    "DraftReportDigest",
    "FinalHandoffCheck",
    "GenerateTaskImplementations",
    "GenerateTaskTests",
    "GenerateWorkflowSkeleton",
    "HumanReview",
    "IngestReport",
    "PlanLLMTask",
    "PlanTask",
    "ValidateWorkspace",
]


_LOG = get_logger(__name__)


# ── Pipeline-step labelling ─────────────────────────────────────────────────
#
# Stable execution order of the 13 PlanMode pipeline nodes —
# discovery (DraftCapabilityNeeds + DiscoverCapabilities) sits
# between DraftImplementationPlan and IR compilation so the IR is
# typed from real evidence. Lives here rather than in :mod:`_pipeline`
# to keep ``tasks.py`` the single source of node-name truth — the
# builder reads class names, this tuple lists them in topological
# order so logs can show ``[plan-node 5/13 ...]``.
_PIPELINE_ORDER: tuple[str, ...] = PLAN_PIPELINE_ORDER
_PIPELINE_TOTAL = len(_PIPELINE_ORDER)
_PIPELINE_INDEX: dict[str, int] = {n: i for i, n in enumerate(_PIPELINE_ORDER, 1)}


def _tag(name: str) -> str:
    """Format the ``[plan-node N/13 NodeName]`` prefix for a log line."""
    return f"[plan-debug {format_node_label(name)}]"


# ── Shared LLM guidance ────────────────────────────────────────────────────


_DISCOVERY_FIRST_RULE = (
    "Capability-discovery-first rule (binding).\n\n"
    "Before choosing implementation details, Task inputs / outputs, or "
    "project-specific types, you MUST rely on capability evidence "
    "returned by the project's capability-discovery stage. Use the "
    "evidence to decide:\n"
    "  - which existing capability implements the step;\n"
    "  - what inputs that capability requires;\n"
    "  - what output it produces;\n"
    "  - which TaskIO types should be declared.\n\n"
    "Do not guess project symbols, APIs, types, services, or helper "
    "names. Primitive payloads may use Python built-ins (str, int, "
    "float, bool, bytes, Path, list[...], dict[...]). Project-specific "
    "objects require evidence-backed symbols drawn from the capability "
    "evidence batch.\n\n"
    "If evidence is missing or insufficient for a step, mark it as "
    "'discovery_required' (or 'unresolved' / 'stubbed' where the schema "
    "allows) instead of inventing a plausible implementation. The "
    "repair loop will re-run discovery for those steps; an invented "
    "name burns budget and forces a rejection."
)


# ── Bases ──────────────────────────────────────────────────────────────────


# Terminal nodes that drive their own review interaction.  The PlanTask
# template method skips the step-policy hook for these so the user
# does not see a redundant "approve HumanReview?" prompt right after
# the actual final-review walkthrough.
_FINAL_STEP_NAMES: frozenset[str] = frozenset({"HumanReview", "FinalHandoffCheck"})


class PlanTask(Task):
    """Base for every task inside the materialize-to-workspace pipeline.

    Implements the per-step review hook as a template method: subclasses
    override :meth:`_execute` to do the work, and the base's
    :meth:`execute` wraps each invocation with a
    :class:`~molexp.agent.review.ReviewPolicy` consultation.

    The hook fires *after* ``_execute`` returns and before the result is
    propagated downstream.  If
    ``ctx.deps.step_policy.review(StepView(...))`` returns
    ``approved=False``, the base raises :class:`StepRejected` carrying
    the view + decision so the repair loop driver can re-run the
    rejected step.  ``approved=True`` (the default
    :class:`~molexp.agent.review.BypassPolicy` answer) returns the
    result verbatim — the wrapper is invisible.

    Terminal nodes (``HumanReview`` / ``FinalHandoffCheck``) drive their
    own review interaction through ``ctx.deps.final_policy``; the
    step-policy hook is skipped for them so the user does not see a
    redundant prompt at the end of the pipeline.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        node_name = type(self).__name__
        _LOG.info(format_progress_start(node_name, iteration=ctx.deps.repair_iteration))
        try:
            result = await self._execute(ctx)
        except Exception as exc:
            _LOG.error(f"[plan] {format_node_label(node_name)} failed: {type(exc).__name__}: {exc}")
            raise

        # Persist every result to the PlanFolder so resume can
        # reconstruct seed_outputs from disk.
        if isinstance(result, BaseModel):
            ctx.deps.plan_folder.write_node_result(node_name, result)

        if node_name in _FINAL_STEP_NAMES:
            ctx.deps.plan_folder.checkpoint(node_name)
            _LOG.info(format_progress_done(node_name, _progress_summary(result)))
            return result

        step_policy = ctx.deps.step_policy
        view = _build_step_view(node_name, result, ctx.deps)
        decision = await step_policy.review(view)
        if decision.approved:
            # Cache the approved output so the repair driver can seed a
            # boundary stub if a later step gets rejected — see
            # _repair_loop._next_round_inputs_from_log.
            ctx.deps.step_outputs_log[node_name] = result
            ctx.deps.plan_folder.checkpoint(node_name)
            _LOG.info(format_progress_done(node_name, _progress_summary(result)))
            return result
        _LOG.info(
            f"[plan] {format_node_label(node_name)} rejected; scheduling repair "
            f"steps={list(decision.target_steps) or [node_name]}"
        )
        raise StepRejected(view, decision)

    async def _execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        """Subclass hook — perform the node's work and return its result.

        Subclasses MUST override this method.  Do not override
        :meth:`execute`: the base method threads the per-step review
        policy around every call.
        """
        raise NotImplementedError


class PlanLLMTask(PlanTask):
    """Plan task that invokes the router to produce structured output.

    The active :class:`~molexp.agent.modes.plan.policy.PlanModelPolicy`
    on ``ctx.deps.policy`` decides which :class:`ModelTier` the
    invocation runs under, keyed by the Task subclass name.
    """

    SYSTEM_PROMPT: ClassVar[str] = ""

    async def invoke_llm[SchemaT: "BaseModel"](
        self,
        ctx: TaskContext[None, PlanDeps, Any],
        *,
        user: str,
        schema: type[SchemaT],
        node_id_suffix: str = "",
    ) -> SchemaT:
        # Per-call logging is owned by the Router (see _pydanticai/router.py).
        # ``node_id_suffix`` lets per-task fanout (codegen) tag each
        # parallel call with its task_id so the router log + usage
        # breakdown attribute work to the right artefact.
        node_name = type(self).__name__
        node_id = f"{node_name}/{node_id_suffix}" if node_id_suffix else node_name
        task_id = node_id_suffix if node_id_suffix else ""
        user = ctx.deps.repair_context.append_to_prompt(
            user,
            node_id=node_id,
            task_id=task_id,
        )
        return await ctx.deps.router.complete_structured(
            tier=ctx.deps.policy.tier_for(node_name),
            system=self.SYSTEM_PROMPT,
            user=user,
            schema=schema,
            node_id=node_id,
        )


def _build_step_view(node_name: str, result: Any, deps: PlanDeps) -> StepView:  # noqa: ANN401
    """Construct a :class:`StepView` for the per-step review hook.

    Extracts a path-bearing summary by walking ``result.model_dump()``
    when the result is a pydantic model, otherwise falls back to a
    minimal view with no artifact paths.  The repair-iteration counter
    is read from :attr:`PlanDeps.repair_iteration` so the reviewer
    sees which round they are in.
    """
    plan_id = deps.plan_folder.plan_id
    summary = f"{node_name} complete"
    artifact_paths: tuple[Path, ...] = ()
    payload: dict[str, Any] | None = None
    if isinstance(result, BaseModel):
        payload = result.model_dump(mode="python")
        artifact_paths = _extract_paths(payload)
    return StepView(
        plan_id=plan_id,
        step_id=node_name,
        summary=summary,
        artifact_paths=artifact_paths,
        payload=payload,
        repair_iteration=deps.repair_iteration,
    )


def _progress_summary(result: Any) -> str:  # noqa: ANN401
    """Return one terse progress suffix for a node result."""
    if not isinstance(result, BaseModel):
        return ""
    fields = result.model_dump(mode="python")
    for key in ("status", "passed", "ready_for_run"):
        if key in fields:
            return f"{key}={fields[key]}"
    for key in ("task_ir_paths", "test_paths", "impl_paths", "evidence", "missing"):
        value = fields.get(key)
        if isinstance(value, (tuple, list)):
            return f"{key}={len(value)}"
    for key in (
        "digest_path",
        "plan_path",
        "workflow_yaml_path",
        "workflow_py_path",
        "report_path",
    ):
        value = fields.get(key)
        if isinstance(value, Path):
            return value.name
    return ""


def _extract_paths(payload: dict[str, Any]) -> tuple[Path, ...]:
    """Walk a result payload and collect every ``Path``-valued field.

    Looks at top-level values and one level of tuples/lists.  Skips
    nested dicts because every PlanMode ``*Result`` keeps paths at the
    top level — adding deeper recursion would surface unrelated paths
    from embedded contracts.
    """
    paths: list[Path] = []
    for value in payload.values():
        if isinstance(value, Path):
            paths.append(value)
        elif isinstance(value, (list, tuple)):
            paths.extend(item for item in value if isinstance(item, Path))
    return tuple(paths)


# ── Node 1: IngestReport ───────────────────────────────────────────────────


class IngestReport(PlanTask):
    """Materialize the user-supplied report verbatim into ``report/original.md``.

    No LLM call. Hashes the input via SHA-256 so downstream nodes can
    detect re-runs against the same content (idempotency hint, not an
    invariant).
    """

    async def _execute(self, ctx: TaskContext[None, PlanDeps, None]) -> IngestReportResult:
        user_input = ctx.config.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("IngestReport requires a non-empty 'user_input' in config.")
        _LOG.debug(f"{_tag('IngestReport')} start chars={len(user_input)}")

        report_dir = ctx.deps.plan_folder.report_dir()
        report_path = report_dir / "original.md"
        _write_text(report_path, user_input)

        digest_hash = hashlib.sha256(user_input.encode("utf-8")).hexdigest()
        _LOG.debug(f"{_tag('IngestReport')} done path={report_path} sha256={digest_hash[:12]}")
        return IngestReportResult(report_path=report_path, report_hash=digest_hash)


# ── Node 2: DraftReportDigest ──────────────────────────────────────────────


class DraftReportDigest(PlanLLMTask):
    """Distill the original report into a structured + markdown digest."""

    SYSTEM_PROMPT = (
        "You are summarizing an experimental report. Return a structured "
        "ReportDigest covering goal, assumptions, systems, expected outputs, "
        "and any missing information."
    )

    async def _execute(self, ctx: TaskContext[None, PlanDeps, IngestReportResult]) -> DigestResult:
        ingest = ctx.inputs
        _LOG.debug(f"{_tag('DraftReportDigest')} start report_path={ingest.report_path}")
        report_text = Path(ingest.report_path).read_text(encoding="utf-8")
        digest = await self.invoke_llm(ctx, user=report_text, schema=ReportDigest)

        digest_path = ctx.deps.plan_folder.report_dir() / "digest.md"
        _write_text(digest_path, _render_digest_markdown(digest))
        goal_chars = len(digest.experimental_goal or "")
        _LOG.debug(f"{_tag('DraftReportDigest')} done path={digest_path} goal_chars={goal_chars}")
        return DigestResult(digest_path=digest_path, digest=digest)


# ── Node 3: DraftImplementationPlan ────────────────────────────────────────


class DraftImplementationPlan(PlanLLMTask):
    """Render a natural-language implementation plan from the digest."""

    SYSTEM_PROMPT = (
        "Given the ReportDigest, draft a PlanBrief covering an overview, "
        "the chosen experimental method with rationale, and an ordered "
        "list of stages.\n\n"
        "Stages are natural-language phrases describing what each step "
        "achieves, not which library to call. Capability discovery runs "
        "in the next stage and binds each step to concrete project "
        "capabilities — do not commit to specific symbol names, class "
        "names, or library APIs here. If you know a step has no project "
        "support today, say so in 'rationale' so the discovery stage can "
        "record the gap."
    )

    async def _execute(self, ctx: TaskContext[None, PlanDeps, DigestResult]) -> PlanBriefResult:
        digest_result = ctx.inputs
        _LOG.debug(
            f"{_tag('DraftImplementationPlan')} start digest_path={digest_result.digest_path}"
        )
        plan_brief = await self.invoke_llm(
            ctx,
            user=digest_result.digest.model_dump_json(),
            schema=PlanBrief,
        )
        plan_path = ctx.deps.plan_folder.plan_dir() / "implementation_plan.md"
        _write_text(plan_path, _render_plan_brief_markdown(plan_brief))
        _LOG.debug(
            f"{_tag('DraftImplementationPlan')} done path={plan_path} "
            f"stages={len(plan_brief.stages)}"
        )
        return PlanBriefResult(plan_path=plan_path, plan_brief=plan_brief)


# ── Node 4: CompileWorkflowIR ──────────────────────────────────────────────


class CompileWorkflowIR(PlanLLMTask):
    """Compile the plan brief into a typed :class:`WorkflowContract`."""

    SYSTEM_PROMPT = (
        _DISCOVERY_FIRST_RULE + "\n\nTranslate the PlanBrief into a WorkflowContract using the "
        "supplied capability evidence batch as the source of truth for "
        "every project-specific type:\n"
        "  - list every task (task_io with inputs / outputs / artifacts);\n"
        "  - declare dependencies via the 'source' field on each input;\n"
        "  - list the validation_checks you want enforced.\n\n"
        "Derive each input / output 'type' field from the evidence:\n"
        "  - primitive payloads use Python built-ins (str, int, float, "
        "    bool, bytes, Path);\n"
        "  - project-specific payloads use the api_ref of the evidence "
        "    row that documents the producing / consuming symbol;\n"
        "  - when no evidence row supports a step's type, set the type "
        "    field to 'discovery_required' instead of inventing one.\n\n"
        "When the evidence batch is empty or marked discovery_skipped, "
        "stick to primitives and 'discovery_required' sentinels — never "
        "fabricate project symbols."
    )

    async def _execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> WorkflowIRResult:
        plan_result = _expect_input(ctx.inputs, "DraftImplementationPlan", PlanBriefResult)
        evidence_batch = _expect_input(ctx.inputs, "DiscoverCapabilities", CapabilityEvidenceBatch)
        _LOG.debug(
            f"{_tag('CompileWorkflowIR')} start plan_path={plan_result.plan_path} "
            f"evidence={len(evidence_batch.evidence)} skipped={evidence_batch.discovery_skipped}"
        )
        contract = await self.invoke_llm(
            ctx,
            user=_augment_user_with_ir_evidence(
                plan_result.plan_brief.model_dump_json(), evidence_batch
            ),
            schema=WorkflowContract,
        )
        ir_path = ctx.deps.plan_folder.ir_dir() / "workflow.yaml"
        contract_dict = default_compiler.contract_to_dict(contract)
        _write_text(ir_path, default_compiler.ir_to_yaml(contract_dict))
        _LOG.debug(f"{_tag('CompileWorkflowIR')} done path={ir_path} tasks={len(contract.task_io)}")
        return WorkflowIRResult(workflow_yaml_path=ir_path, contract=contract)


# ── Node 5: CompileTaskIR ──────────────────────────────────────────────────


class CompileTaskIR(PlanLLMTask):
    """Compile a per-task brief for every task in the workflow contract.

    Empty-task contracts (``contract.task_io == ()``) emit an empty
    ``task_ir_paths`` tuple — sub-spec 01 explicitly permits empty
    contracts, so this node is not allowed to fail on them.
    """

    SYSTEM_PROMPT = (
        _DISCOVERY_FIRST_RULE + "\n\nGiven one TaskIO entry from a WorkflowContract plus the "
        "capability evidence batch, draft a TaskIRBrief covering its "
        "responsibility, success criteria, failure conditions, and "
        "minimal test expectations.\n\n"
        "'responsibility' MUST name the evidenced symbol the task uses "
        "(api_ref drawn from the evidence batch). 'success_criteria' "
        "and 'test_expectations' should assert on the documented return "
        "shape of that symbol. Set is_stub=true when no evidence row "
        "supports the task — never invent a symbol to fill the gap."
    )

    async def _execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> TaskIRResult:
        ir_result = _expect_input(ctx.inputs, "CompileWorkflowIR", WorkflowIRResult)
        evidence_batch = _expect_input(ctx.inputs, "DiscoverCapabilities", CapabilityEvidenceBatch)
        tasks_ir_dir = ctx.deps.plan_folder.tasks_ir_dir()
        task_ios = ir_result.contract.task_io
        n_tasks = len(task_ios)
        tier = ctx.deps.policy.tier_for("CompileTaskIR")
        _LOG.debug(
            f"{_tag('CompileTaskIR')} start tasks={n_tasks} (parallel) "
            f"evidence={len(evidence_batch.evidence)} skipped={evidence_batch.discovery_skipped}"
        )

        async def _draft_one(task_io: TaskIO) -> TaskIRBrief:
            user = _augment_user_with_ir_evidence(task_io.model_dump_json(), evidence_batch)
            user = ctx.deps.repair_context.append_to_prompt(
                user,
                node_id=f"CompileTaskIR/{task_io.task_id}",
                task_id=task_io.task_id,
            )
            brief = await ctx.deps.router.complete_structured(
                tier=tier,
                system=self.SYSTEM_PROMPT,
                user=user,
                schema=TaskIRBrief,
                node_id=f"CompileTaskIR/{task_io.task_id}",
            )
            # Force the brief's task_id to match the contract entry —
            # the LLM may forget; the workflow_contract is authoritative.
            if brief.task_id != task_io.task_id:
                brief = brief.model_copy(update={"task_id": task_io.task_id})
            return brief

        briefs_seq: tuple[TaskIRBrief, ...] = (
            tuple(await asyncio.gather(*[_draft_one(t) for t in task_ios])) if task_ios else ()
        )

        paths: list[Path] = []
        briefs: list[TaskIRBrief] = []
        for task_io, brief in zip(task_ios, briefs_seq, strict=True):
            task_yaml = tasks_ir_dir / f"{task_io.task_id}.yaml"
            _write_text(
                task_yaml,
                yaml.safe_dump(
                    brief.model_dump(mode="json"),
                    sort_keys=False,
                    default_flow_style=False,
                ),
            )
            paths.append(task_yaml)
            briefs.append(brief)

        n_stub = sum(1 for b in briefs if b.is_stub)
        _LOG.debug(f"{_tag('CompileTaskIR')} done briefs={len(briefs)} stubs={n_stub}")
        return TaskIRResult(task_ir_paths=tuple(paths), briefs=tuple(briefs))


# ── Node 6: GenerateWorkflowSkeleton ───────────────────────────────────────


class GenerateWorkflowSkeleton(PlanTask):
    """Emit the experiment Python package skeleton.

    Templated, not LLM-driven — the contract is structured enough to
    deterministically generate ``src/experiment/{__init__.py,
    workflow.py, tasks/__init__.py}``. Each generated file is
    syntax-checked via :func:`compile`; a failure raises
    :class:`SkeletonCompileError`. Per-task module bodies are out of
    scope here (sub-spec 06's ``GenerateTaskImplementations`` writes
    them).
    """

    async def _execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> SkeletonResult:
        # Read three upstreams: the workflow contract for topology, the
        # task IR result to confirm the per-task IR has been written,
        # and the capability evidence batch for chain ordering.
        # Skeleton output is templated (not LLM-driven), so the evidence
        # gate's source-AST diff doesn't apply here — the chain
        # dependency exists only so codegen happens after discovery.
        ir_result = _expect_input(ctx.inputs, "CompileWorkflowIR", WorkflowIRResult)
        _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        _expect_input(ctx.inputs, "DiscoverCapabilities", CapabilityEvidenceBatch)
        _LOG.debug(
            f"{_tag('GenerateWorkflowSkeleton')} start tasks={len(ir_result.contract.task_io)}"
        )

        contract = ir_result.contract
        package_path = ctx.deps.plan_folder.experiment_pkg_dir()
        tasks_pkg = ctx.deps.plan_folder.tasks_pkg_dir()

        # __init__.py (top-level package)
        package_init = package_path / "__init__.py"
        _validate_python(_PACKAGE_INIT_BODY, str(package_init))
        _write_text(package_init, _PACKAGE_INIT_BODY)

        # workflow.py
        workflow_py = package_path / "workflow.py"
        workflow_source = _render_workflow_module(contract)
        _validate_python(workflow_source, str(workflow_py))
        _write_text(workflow_py, workflow_source)

        # tasks/__init__.py
        tasks_init = tasks_pkg / "__init__.py"
        _validate_python(_TASKS_INIT_BODY, str(tasks_init))
        _write_text(tasks_init, _TASKS_INIT_BODY)

        _LOG.debug(
            f"{_tag('GenerateWorkflowSkeleton')} done workflow_py={workflow_py} pkg={package_path}"
        )
        return SkeletonResult(workflow_py_path=workflow_py, package_path=package_path)


# ── Node 7: GenerateTaskTests ──────────────────────────────────────────────


class GenerateTaskTests(PlanLLMTask):
    """Generate one pytest module per task plus a topology-pin test.

    Reads :class:`TaskIRResult` (briefs) and :class:`WorkflowIRResult`
    (the contract). For each brief, asks the provider for one
    :class:`TaskTestModule` and writes it to ``tests/test_<task>.py``.
    A separate ``tests/test_workflow_structure.py`` asserts the
    generated workflow's topology against the IR.
    """

    SYSTEM_PROMPT = (
        _DISCOVERY_FIRST_RULE + "\n\nGiven a TaskIRBrief plus its TaskIO declaration, draft a "
        "pytest module that exercises the task. If is_stub=True, emit a "
        "single test that calls pytest.skip('stub'). Otherwise, write at "
        "least one happy-path test referencing the documented inputs / "
        "outputs.\n\n"
        "Tests MUST import only from project namespaces named in the "
        "capability evidence batch plus stdlib. Use the documented "
        "symbols, signatures, and return shapes verbatim — construct "
        "fixtures from evidenced types, run the task, assert the result "
        "matches the documented type and attributes. Do not fabricate "
        "symbol names that are not in the evidence batch."
    )

    async def _execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> TaskTestsResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        _expect_input(ctx.inputs, "GenerateWorkflowSkeleton", SkeletonResult)
        evidence_batch = _expect_input(ctx.inputs, "DiscoverCapabilities", CapabilityEvidenceBatch)

        handle = ctx.deps.plan_folder
        repair_targets = ctx.deps.repair_target_tasks
        _targets_repr = list(repair_targets) if repair_targets else None
        _LOG.debug(
            f"{_tag('GenerateTaskTests')} start briefs={len(ir_result.briefs)} "
            f"repair_targets={_targets_repr} "
            f"evidence={len(evidence_batch.evidence)} skipped={evidence_batch.discovery_skipped}"
        )

        async def render_real(brief: TaskIRBrief) -> str:
            module = await self.invoke_llm(
                ctx,
                user=_augment_user_with_evidence(brief.model_dump_json(), evidence_batch),
                schema=TaskTestModule,
                node_id_suffix=brief.task_id,
            )
            if module.task_id != brief.task_id:
                module = module.model_copy(update={"task_id": brief.task_id})
            _gate_evidence(
                source=module.source,
                evidence_refs=module.evidence_refs,
                batch=evidence_batch,
                is_stub=False,
                where=f"GenerateTaskTests/{brief.task_id}",
            )
            return module.source

        paths = await _per_brief_codegen(
            briefs=ir_result.briefs,
            repair_targets=repair_targets,
            existing_path_for=lambda task_id: handle.tests_dir() / f"test_{task_id}.py",
            render_stub=_render_stub_test_module,
            render_real=render_real,
            write=handle.write_test_module,
        )

        # Topology-pin test always lands.
        handle.write_workflow_structure_test(_render_workflow_structure_test("ir/workflow.yaml"))

        _LOG.debug(f"{_tag('GenerateTaskTests')} done test_modules={len(paths)}")
        return TaskTestsResult(test_paths=paths)


# ── Node 8: GenerateTaskImplementations ────────────────────────────────────


class GenerateTaskImplementations(PlanLLMTask):
    """Generate one runnable module per task in the workflow contract.

    For each :class:`TaskIRBrief` whose ``is_stub`` is False, asks
    the provider for a :class:`TaskImplementationModule` and writes
    the LLM-emitted source verbatim. For stubs, writes a tiny module
    body that ``raise NotImplementedError(<reason>)`` — sub-spec 06's
    v1 stub-tolerance contract.
    """

    SYSTEM_PROMPT = (
        _DISCOVERY_FIRST_RULE + "\n\nGiven a TaskIRBrief plus its TaskIO declaration, write a "
        "Python module implementing the task as a molexp.workflow.Task "
        "subclass with `async def execute(ctx)`. The body MUST call into "
        "symbols listed in the capability evidence batch — never write a "
        "hand-rolled algorithm when an evidenced function exists. Set "
        "is_stub=true when no evidence row in the batch can satisfy "
        "this task; the stub will raise NotImplementedError and the "
        "repair loop will re-run discovery for the missing capability."
    )

    async def _execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> TaskImplementationsResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        _expect_input(ctx.inputs, "GenerateWorkflowSkeleton", SkeletonResult)
        evidence_batch = _expect_input(ctx.inputs, "DiscoverCapabilities", CapabilityEvidenceBatch)

        handle = ctx.deps.plan_folder
        repair_targets = ctx.deps.repair_target_tasks
        _targets_repr = list(repair_targets) if repair_targets else None
        _LOG.debug(
            f"{_tag('GenerateTaskImplementations')} start briefs={len(ir_result.briefs)} "
            f"repair_targets={_targets_repr} "
            f"evidence={len(evidence_batch.evidence)} skipped={evidence_batch.discovery_skipped}"
        )

        async def render_real(brief: TaskIRBrief) -> str:
            module = await self.invoke_llm(
                ctx,
                user=_augment_user_with_evidence(brief.model_dump_json(), evidence_batch),
                schema=TaskImplementationModule,
                node_id_suffix=brief.task_id,
            )
            if module.is_stub:
                # Stubs ship empty `__capability_evidence__: () = ()` and
                # skip the AST diff entirely (spec: stubs allowed to
                # raise NotImplementedError; no external API calls).
                return _render_stub_implementation_module(brief.task_id)
            _gate_evidence(
                source=module.source,
                evidence_refs=module.evidence_refs,
                batch=evidence_batch,
                is_stub=False,
                where=f"GenerateTaskImplementations/{brief.task_id}",
            )
            return module.source

        paths = await _per_brief_codegen(
            briefs=ir_result.briefs,
            repair_targets=repair_targets,
            existing_path_for=lambda task_id: handle.tasks_pkg_dir() / f"{task_id}.py",
            render_stub=_render_stub_implementation_module,
            render_real=render_real,
            write=handle.write_task_implementation,
        )

        _LOG.debug(f"{_tag('GenerateTaskImplementations')} done impl_modules={len(paths)}")
        return TaskImplementationsResult(impl_paths=paths)


# ── Node 9: ValidateWorkspace ──────────────────────────────────────────────


class ValidateWorkspace(PlanTask):
    """Run the deterministic validation pass over the materialized workspace.

    The pass includes RunMode-style import validation and generic
    workflow contract validation. Results land in both
    ``validation_report.md`` and ``validation_report.yaml``.
    """

    async def _execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> ValidationResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        # GenerateTaskImplementations / GenerateTaskTests outputs are
        # validated through the on-disk artifacts; pull them only to
        # ensure the upstream order ran.
        _expect_input(ctx.inputs, "GenerateTaskTests", TaskTestsResult)
        _expect_input(ctx.inputs, "GenerateTaskImplementations", TaskImplementationsResult)
        _LOG.debug(f"{_tag('ValidateWorkspace')} start")

        handle = ctx.deps.plan_folder
        checks = _run_workspace_validation(
            handle,
            task_ir=ir_result,
            module_name="experiment.workflow",
            symbol_name="create_workflow",
        )
        passed = _checks_passed(checks)
        status = "ready_for_review" if passed else "validation_failed"
        summary = _summarize_checks(checks)
        failed_names = [c.name for c in checks if not c.passed]
        if passed:
            _LOG.debug(f"{_tag('ValidateWorkspace')} passed checks={len(checks)}")
        else:
            _LOG.warning(
                f"{_tag('ValidateWorkspace')} failed checks={len(checks)} failures={failed_names}"
            )
        report = ValidationReport(passed=passed, checks=tuple(checks), summary=summary)
        report_path = handle.write_validation_report(report)
        # Preserve repair_iterations / repair_history from any prior manifest
        # so the review→repair driver's accumulating state is not clobbered
        # when ValidateWorkspace re-writes the manifest each iteration.
        existing = _load_manifest_from_disk(handle.manifest_path())
        manifest = (
            existing
            if existing is not None
            else _build_manifest_stub(handle.plan_id, handle.ir_dir() / "workflow.yaml")
        )
        manifest = manifest.model_copy(
            update={
                "task_ir_paths": tuple(sorted(handle.tasks_ir_dir().glob("*.yaml"))),
                "model_policy_snapshot": ctx.deps.policy.model_dump(mode="json"),
                "status": status,
            }
        )
        handle.write_manifest(manifest)
        return ValidationResult(
            report_path=report_path,
            passed=passed,
            summary=summary,
            checks=tuple(checks),
            status=status,
        )


# ── Node 10: HumanReview ───────────────────────────────────────────────────


class HumanReview(PlanTask):
    """Gate the plan with the final-review :class:`~molexp.agent.review.ReviewPolicy`.

    Human approval is deliberately separate from RunMode readiness.
    The next node, ``FinalHandoffCheck``, is the only node allowed to
    mark the workspace ``ready_for_run``.
    """

    async def _execute(self, ctx: TaskContext[None, PlanDeps, ValidationResult]) -> HandoffResult:
        validation_result = ctx.inputs
        _LOG.debug(
            f"{_tag('HumanReview')} start iter={ctx.deps.repair_iteration} "
            f"validation_passed={validation_result.passed}"
        )

        # Re-read upstream artefacts from the workspace to assemble the
        # review view + handoff. We pull from disk rather than threading
        # them all via ctx.inputs to keep the workflow fan-in shallow.
        handle = ctx.deps.plan_folder
        digest = await _read_yaml_into(
            handle.report_dir() / "digest.md", ReportDigest, fallback="digest"
        )
        plan_brief = await _read_yaml_into(
            handle.plan_dir() / "implementation_plan.md",
            PlanBrief,
            fallback="plan",
        )
        contract_yaml = handle.ir_dir() / "workflow.yaml"
        contract_dict = yaml.safe_load(contract_yaml.read_text())
        contract = default_compiler.dict_to_contract(contract_dict)

        plan_id = handle.plan_id
        # Surface failed-check identifiers for the review view so the
        # reviewer can target the offending tasks via
        # ``ReviewDecision.target_*``. Empty tuple on first pass / when
        # everything passed.
        failures = tuple(check.name for check in validation_result.checks if not check.passed)
        view = PlanReviewView(
            plan_id=plan_id,
            experiment_workspace_path=handle.root(),
            digest=digest,
            plan_brief=plan_brief,
            contract=contract,
            validation_passed=validation_result.passed,
            validation_summary=validation_result.summary,
            previous_validation_failures=failures,
            repair_iteration=ctx.deps.repair_iteration,
        )

        decision = await ctx.deps.final_policy.review(view)
        _LOG.debug(
            f"{_tag('HumanReview')} decision approved={decision.approved} "
            f"target_steps={list(decision.target_steps)} "
            f"target_tasks={list(decision.target_task_ids)} "
            f"cascade={decision.cascade_downstream}"
        )

        # Build the handoff regardless — both branches consume it.
        existing_manifest = _load_manifest_from_disk(handle.manifest_path())
        validation_report = ValidationReport(
            passed=validation_result.passed,
            summary=validation_result.summary,
        )
        new_status = _review_status(
            decision=decision,
            validation_passed=validation_result.passed,
        )
        manifest_for_handoff = (
            existing_manifest or _build_manifest_stub(plan_id, contract_yaml)
        ).model_copy(update={"status": new_status})
        handoff = PlanRunHandoff(
            plan_id=plan_id,
            experiment_workspace_path=handle.root(),
            workflow_yaml_path=contract_yaml,
            source_root=Path("src"),
            task_ir_paths=tuple(handle.tasks_ir_dir().glob("*.yaml"))
            if handle.tasks_ir_dir().exists()
            else (),
            entrypoint_module="experiment.workflow",
            entrypoint_symbol="create_workflow",
            manifest_snapshot=manifest_for_handoff,
            validation_report_snapshot=validation_report,
            created_at=_utcnow(),
        )

        _persist_manifest_with_handoff(
            handle,
            manifest_for_handoff,
            handoff,
            decision=decision,
            ready_for_run=False,
            validation_passed=validation_result.passed,
            validation_summary=validation_result.summary,
            status=new_status,
        )

        _LOG.debug(f"{_tag('HumanReview')} done status={new_status}")
        return HandoffResult(
            handoff=handoff,
            decision=decision,
            validation_passed=validation_result.passed,
            ready_for_run=False,
            status=new_status,
            report_path=validation_result.report_path,
            validation_checks=validation_result.checks,
        )


# ── Node 11: FinalHandoffCheck ─────────────────────────────────────────────


class FinalHandoffCheck(PlanTask):
    """Run the final RunMode-style load and contract validation gate."""

    async def _execute(self, ctx: TaskContext[None, PlanDeps, HandoffResult]) -> HandoffResult:
        prior = ctx.inputs
        handle = ctx.deps.plan_folder
        _LOG.debug(
            f"{_tag('FinalHandoffCheck')} start prior_status={prior.status} "
            f"approved={prior.decision.approved}"
        )
        final_checks = _run_handoff_validation(handle, prior.handoff, prefix="final_")
        checks = (*prior.validation_checks, *final_checks)
        validation_passed = prior.validation_passed and _checks_passed(final_checks)
        ready_for_run = bool(prior.decision.approved and validation_passed)
        status = _final_status(
            decision=prior.decision,
            validation_passed=validation_passed,
            ready_for_run=ready_for_run,
        )
        failed_final = [c.name for c in final_checks if not c.passed]
        if validation_passed:
            _LOG.debug(
                f"{_tag('FinalHandoffCheck')} done status={status} ready_for_run={ready_for_run}"
            )
        else:
            _LOG.warning(
                f"{_tag('FinalHandoffCheck')} failed status={status} failures={failed_final}"
            )
        summary = _summarize_checks(checks)
        report = ValidationReport(passed=validation_passed, checks=checks, summary=summary)
        report_path = handle.write_validation_report(report)

        manifest = prior.handoff.manifest_snapshot.model_copy(update={"status": status})
        handoff = prior.handoff.model_copy(
            update={
                "manifest_snapshot": manifest,
                "validation_report_snapshot": report,
            }
        )
        _persist_manifest_with_handoff(
            handle,
            manifest,
            handoff,
            decision=prior.decision,
            ready_for_run=ready_for_run,
            validation_passed=validation_passed,
            validation_summary=summary,
            status=status,
        )
        return prior.model_copy(
            update={
                "handoff": handoff,
                "validation_passed": validation_passed,
                "ready_for_run": ready_for_run,
                "status": status,
                "report_path": report_path,
                "validation_checks": checks,
            }
        )


# ── Helpers (extended for sub-spec 06) ─────────────────────────────────────


def _augment_user_with_ir_evidence(user: str, batch: CapabilityEvidenceBatch) -> str:
    """Append a YAML-rendered evidence block to an IR-compilation prompt.

    Used by ``CompileWorkflowIR`` and ``CompileTaskIR`` — both decide
    project-specific TaskIO types from the discovery batch, but they
    are *not* writing executable modules, so they do not need the
    ``__capability_evidence__`` literal contract that codegen nodes
    obey (that lives in :func:`_augment_user_with_evidence`).

    When ``batch.discovery_skipped`` is True or the batch carries no
    evidence rows, returns ``user`` plus a short note instructing the
    LLM to stick to primitive types and ``"discovery_required"``
    sentinels. The discovery-first rule is already in the system
    prompt; the appendix is the binding evidence list.
    """
    if batch.discovery_skipped or not batch.evidence:
        return (
            f"{user}\n\n"
            "## Discovered project evidence\n\n"
            "No evidence rows are available for this step. Use Python "
            "primitives for primitive payloads and the sentinel string "
            '"discovery_required" for any project-specific type. Do '
            "not invent project symbols, classes, or helper names."
        )
    evidence_payload = [
        {
            "api_ref": e.api_ref,
            "signature": e.signature,
            "doc_summary": e.doc_summary,
        }
        for e in batch.evidence
    ]
    appendix = yaml.safe_dump(
        {"discovered_evidence": evidence_payload},
        sort_keys=False,
        default_flow_style=False,
    )
    return (
        f"{user}\n\n"
        "## Discovered project evidence (binding)\n\n"
        "Project-specific TaskIO types MUST reference one of the "
        "api_ref values listed below. Primitive payloads use Python "
        "built-ins. For steps with no matching evidence row, set the "
        'type field to "discovery_required" — do not invent a '
        "replacement.\n\n"
        f"{appendix}"
    )


def _augment_user_with_evidence(user: str, batch: CapabilityEvidenceBatch) -> str:
    """Append a YAML-rendered evidence block to a per-brief LLM prompt.

    When ``batch.discovery_skipped`` is True, returns ``user`` verbatim
    — pure-stdlib codegen paths do not see an evidence appendix and
    are exempt from the ``__capability_evidence__`` literal requirement.

    Otherwise produces a structured appendix listing each evidence
    row's ``api_ref`` / ``signature`` / ``doc_summary`` and instructs
    the LLM that:

    1. it may only reference ``api_ref`` values that appear in the
       appendix;
    2. the generated module's top level **must** contain a literal
       ``__capability_evidence__: tuple[str, ...] = (...)`` whose
       contents (parseable by :func:`ast.literal_eval`) match the
       schema field ``evidence_refs`` exactly.

    The strict matching is checked downstream by
    :func:`_gate_evidence` immediately after the LLM returns and again
    by :func:`~molexp.agent.modes.plan.capability.validate_codegen_evidence`
    after the source is written to disk.
    """
    if batch.discovery_skipped:
        return user
    evidence_payload = [
        {
            "api_ref": e.api_ref,
            "signature": e.signature,
            "doc_summary": e.doc_summary,
        }
        for e in batch.evidence
    ]
    hints_payload = [
        {
            "namespace": hint.namespace,
            "strength": hint.strength,
            "query_hints": list(hint.query_hints),
            "reason": hint.reason,
        }
        for hint in batch.hints
    ]
    appendix = yaml.safe_dump(
        {
            "discovered_capability_evidence": evidence_payload,
            "discovery_hints": hints_payload,
            "fallback_reasons": list(batch.fallback_reasons),
        },
        sort_keys=False,
        default_flow_style=False,
    )
    return (
        f"{user}\n\n"
        "## Discovered capability evidence (binding)\n\n"
        "You may reference only the api_ref values listed below.\n\n"
        "Generated module REQUIREMENTS:\n"
        "1. Set the schema's `evidence_refs` field to the api_ref values you actually used.\n"
        "2. Emit a module-level literal:\n"
        "       __capability_evidence__: tuple[str, ...] = (\n"
        '           "<api_ref_1>",\n'
        "           ...,\n"
        "       )\n"
        "   It MUST be a top-level assignment (not a comment / docstring / dict),\n"
        "   parseable by ast.literal_eval, and the set of refs MUST equal `evidence_refs`.\n\n"
        f"{appendix}"
    )


def _gate_evidence(
    *,
    source: str,
    evidence_refs: tuple[str, ...],
    batch: CapabilityEvidenceBatch,
    is_stub: bool,
    where: str,
) -> None:
    """Run the two-stage evidence gate on a freshly generated module.

    Stage 1 — declared-block consistency: assert the source's
    ``__capability_evidence__: tuple[str, ...] = (...)`` literal
    matches the schema's ``evidence_refs`` field. Mismatch raises
    ``UnevidencedApiReference(reason='declared_block_mismatch')``.

    Stage 2 — AST diff against discovery batch: invoke
    :func:`~molexp.agent.modes.plan.capability.validate_codegen_evidence`
    and raise ``UnevidencedApiReference`` (no special reason slug) on
    any non-empty miss set. The validator's own three-way diff is the
    source of truth for the missing-ref detail.

    Stub paths (``is_stub=True``) and ``batch.discovery_skipped=True``
    short-circuit both stages — the spec exempts them from the gate.

    ``where`` is included in error messages so the repair loop can
    log which (node, task) triggered the failure.
    """
    if is_stub or batch.discovery_skipped:
        return

    declared = _extract_declared_block_refs(source)
    if set(declared) != set(evidence_refs):
        diff = sorted(set(declared) ^ set(evidence_refs))
        raise UnevidencedApiReference(
            f"{where}: declared_block_mismatch — source's __capability_evidence__ "
            f"({sorted(declared)}) does not equal schema.evidence_refs "
            f"({sorted(evidence_refs)}).",
            refs=tuple(diff),
            reason="declared_block_mismatch",
            detail=f"node={where}",
        )

    misses = validate_codegen_evidence(source, batch)
    if misses:
        miss_refs = tuple(_pick_ref(m.detail) for m in misses if _pick_ref(m.detail))
        raise UnevidencedApiReference(
            f"{where}: post-codegen AST validator returned {len(misses)} miss(es).",
            refs=miss_refs,
            reason="",
            detail="; ".join(m.detail for m in misses),
        )


def _extract_declared_block_refs(source: str) -> set[str]:
    """Pull the ``__capability_evidence__`` literal out of ``source``.

    Returns the empty set when the literal is absent or malformed —
    :func:`_gate_evidence` then surfaces the mismatch against the
    schema's ``evidence_refs`` if any were declared.

    Defers to :func:`molexp.agent.modes.plan.capability.extract_declared_refs`
    rather than re-implementing the AST walk so both call sites share
    one extraction surface.
    """
    import ast as _ast

    from molexp.agent.modes.plan.capability import extract_declared_refs

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return set()
    return extract_declared_refs(tree)


def _pick_ref(detail: str) -> str:
    """Best-effort extraction of an api_ref from a MissingCapability.detail.

    The validator formats details like ``"foo.bar.Baz not in evidence batch"``
    or ``"foo.bar.Baz used in AST but missing from __capability_evidence__"``;
    in both shapes the api_ref is the first whitespace-separated token.
    """
    return detail.split(" ", 1)[0] if detail else ""


async def _per_brief_codegen[BriefT: TaskIRBrief](
    *,
    briefs: tuple[BriefT, ...],
    repair_targets: tuple[str, ...] | frozenset[str] | None,
    existing_path_for: Callable[[str], Path],
    render_stub: Callable[[str], str],
    render_real: Callable[[BriefT], Awaitable[str]],
    write: Callable[[str, str], Path],
) -> tuple[Path, ...]:
    """Apply the repair-loop filter to a sequence of briefs and codegen
    the surviving ones in parallel.

    For each brief, picks one of three actions:
    - **reuse** — repair-loop filter says "skip" AND the previous round's
      output is on disk: that path is returned verbatim.
    - **stub** — ``brief.is_stub`` is True: ``render_stub`` produces the
      source synchronously (no LLM round-trip).
    - **real** — ``render_real`` produces the source via the provider; all
      "real" calls are gathered concurrently with :func:`asyncio.gather`
      so wall-clock cost is bounded by the slowest call rather than the
      sum.

    The returned tuple preserves brief order so downstream consistency
    checks see the canonical sequence.
    """
    real_indices: list[int] = []
    real_coros: list[Awaitable[str]] = []
    sources: list[str | Path] = [Path()] * len(briefs)
    for i, brief in enumerate(briefs):
        existing = existing_path_for(brief.task_id)
        if repair_targets is not None and brief.task_id not in repair_targets and existing.exists():
            sources[i] = existing
            continue
        if brief.is_stub:
            sources[i] = render_stub(brief.task_id)
            continue
        real_indices.append(i)
        real_coros.append(render_real(brief))

    if real_coros:
        for i, source in zip(real_indices, await asyncio.gather(*real_coros), strict=True):
            sources[i] = source

    paths: list[Path] = []
    for brief, source in zip(briefs, sources, strict=True):
        if isinstance(source, Path):
            paths.append(source)
        else:
            paths.append(write(brief.task_id, source))
    return tuple(paths)


def _render_stub_test_module(task_id: str) -> str:
    return (
        f'"""Generated test for ``{task_id}`` — implementation is a stub."""\n'
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        f"def test_{task_id}_stub() -> None:\n"
        '    pytest.skip("stub")\n'
    )


def _render_workflow_structure_test(ir_yaml_rel: str) -> str:
    return (
        '"""Generated topology-pin test for the experiment workflow."""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import pytest\n"
        "import yaml\n"
        "\n"
        "from molexp.workflow import default_compiler\n"
        "\n"
        "\n"
        "def test_workflow_topology_matches_ir() -> None:\n"
        f"    ir_text = Path(__file__).resolve().parent.parent.joinpath({ir_yaml_rel!r}).read_text()\n"
        "    contract = default_compiler.dict_to_contract(default_compiler.yaml_to_ir(ir_text))\n"
        "    declared_ids = {tio.task_id for tio in contract.task_io}\n"
        "    try:\n"
        "        from experiment.workflow import create_workflow\n"
        "    except ImportError:\n"
        '        pytest.skip("experiment.workflow import failed; skipping topology pin")\n'
        "        return\n"
        "    actual_ids = {t.name for t in create_workflow()._tasks}\n"
        "    assert declared_ids == actual_ids\n"
    )


def _render_stub_implementation_module(task_id: str) -> str:
    return (
        f'"""Stub implementation for ``{task_id}`` — fill in to enable RunMode."""\n'
        "\n"
        "from molexp.workflow import Task\n"
        "\n"
        "\n"
        f"class {_camel_case(task_id)}(Task):\n"
        f'    """Stub for {task_id} — populate this body to make RunMode runnable."""\n'
        "\n"
        "    async def execute(self, ctx) -> None:  # type: ignore[no-untyped-def, override]\n"
        f'        raise NotImplementedError("{task_id} not yet implemented")\n'
    )


def _run_workspace_validation(
    handle: PlanFolder,
    *,
    task_ir: TaskIRResult,
    module_name: str,
    symbol_name: str,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    contract = _load_contract_for_validation(handle, checks)

    for task_yaml in (
        sorted(handle.tasks_ir_dir().glob("*.yaml")) if handle.tasks_ir_dir().exists() else ()
    ):
        try:
            yaml.safe_load(task_yaml.read_text())
            checks.append(
                CheckResult(
                    name=f"task_ir_parseable[{task_yaml.stem}]",
                    passed=True,
                    severity="info",
                )
            )
        except yaml.YAMLError as exc:
            checks.append(
                CheckResult(
                    name=f"task_ir_parseable[{task_yaml.stem}]",
                    passed=False,
                    severity="error",
                    detail=str(exc),
                )
            )

    for brief in task_ir.briefs:
        impl_path = handle.tasks_pkg_dir() / f"{brief.task_id}.py"
        checks.append(
            CheckResult(
                name=f"impl_present[{brief.task_id}]",
                passed=impl_path.exists(),
                severity="info" if impl_path.exists() else "error",
                detail="" if impl_path.exists() else f"missing {impl_path.name}",
            )
        )
        test_path = handle.tests_dir() / f"test_{brief.task_id}.py"
        checks.append(
            CheckResult(
                name=f"test_present[{brief.task_id}]",
                passed=test_path.exists(),
                severity="info" if test_path.exists() else "error",
                detail="" if test_path.exists() else f"missing {test_path.name}",
            )
        )

    for rel in (
        "report/original.md",
        "plan/implementation_plan.md",
        "src/experiment/__init__.py",
        "src/experiment/workflow.py",
    ):
        path = handle.root() / rel
        checks.append(
            CheckResult(
                name=f"path_exists[{rel}]",
                passed=path.exists(),
                severity="info" if path.exists() else "error",
                detail="" if path.exists() else f"missing {rel}",
            )
        )

    workflow_py = handle.experiment_pkg_dir() / "workflow.py"
    if workflow_py.exists():
        try:
            compile(workflow_py.read_text(), str(workflow_py), "exec")
            checks.append(
                CheckResult(
                    name="workflow_module_compiles",
                    passed=True,
                    severity="info",
                )
            )
        except SyntaxError as exc:
            checks.append(
                CheckResult(
                    name="workflow_module_compiles",
                    passed=False,
                    severity="error",
                    detail=f"SyntaxError: {exc.msg} at line {exc.lineno}",
                )
            )

    if contract is not None:
        contract_task_ids = {tio.task_id for tio in contract.task_io}
        task_ir_ids = {brief.task_id for brief in task_ir.briefs}
        mismatch = contract_task_ids.symmetric_difference(task_ir_ids)
        checks.append(
            CheckResult(
                name="task_ir_matches_workflow_ir",
                passed=not mismatch,
                severity="info" if not mismatch else "error",
                detail="" if not mismatch else f"mismatched task ids: {sorted(mismatch)}",
            )
        )
        if not contract.task_io:
            checks.append(
                CheckResult(
                    name="contract_has_tasks",
                    passed=True,
                    severity="warning",
                    detail="no tasks declared in workflow contract",
                )
            )

    # Phase 6 — capability-evidence dual-signal check.
    checks.extend(_capability_evidence_checks(handle))

    handoff = PlanRunHandoff(
        plan_id=handle.plan_id,
        experiment_workspace_path=handle.root(),
        workflow_yaml_path=handle.ir_dir() / "workflow.yaml",
        source_root=Path("src"),
        task_ir_paths=tuple(sorted(handle.tasks_ir_dir().glob("*.yaml")))
        if handle.tasks_ir_dir().exists()
        else (),
        entrypoint_module=module_name,
        entrypoint_symbol=symbol_name,
        manifest_snapshot=_build_manifest_stub(handle.plan_id, handle.ir_dir() / "workflow.yaml"),
        validation_report_snapshot=ValidationReport(passed=False),
        created_at=_utcnow(),
    )
    checks.extend(_run_handoff_validation(handle, handoff, contract=contract))
    return checks


def _capability_evidence_checks(handle: PlanFolder) -> list[CheckResult]:
    """Phase 6 dual-signal capability evidence check.

    For each generated module under ``src/experiment/tasks/*.py`` and
    ``tests/test_*.py``, runs two independent checks against the on-disk
    ``capability/evidence.yaml`` batch:

    1. **declared_refs** — the module-level
       ``__capability_evidence__`` literal MUST list only ``api_ref``
       values that appear in ``evidence.yaml``.
    2. **ast_refs** — every tracked dotted-path reference the AST
       walker discovers MUST also appear in
       ``evidence.yaml``.

    Both signals diff against the *same* evidence-batch ``api_ref`` set,
    but are reported as separate :class:`CheckResult` rows so reviewers
    can tell which signal flagged a miss.

    Special cases:

    * ``evidence.yaml`` missing — emits an `info` check noting the
      capability dir was not materialized yet (the LLM-driven
      :meth:`DiscoverCapabilities` short-circuited on a stub probe).
    * ``discovery_skipped=True`` — both checks pass with detail
      ``"discovery_skipped"`` (pure-stdlib paths exempt from the gate).
    """
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidenceBatch,
        extract_ast_refs,
        extract_declared_refs,
    )

    evidence_path = handle.capability_dir() / "evidence.yaml"
    if not evidence_path.exists():
        return [
            CheckResult(
                name="capability_evidence_present",
                passed=True,
                severity="info",
                detail="evidence.yaml not materialized (probe is null or pipeline aborted)",
            )
        ]

    try:
        batch = CapabilityEvidenceBatch.model_validate(yaml.safe_load(evidence_path.read_text()))
    except (yaml.YAMLError, ValueError) as exc:
        return [
            CheckResult(
                name="capability_evidence_parseable",
                passed=False,
                severity="error",
                detail=f"failed to parse evidence.yaml: {exc}",
            )
        ]

    if batch.discovery_skipped:
        return [
            CheckResult(
                name="capability_evidence_check[declared]",
                passed=True,
                severity="info",
                detail="discovery_skipped",
            ),
            CheckResult(
                name="capability_evidence_check[ast]",
                passed=True,
                severity="info",
                detail="discovery_skipped",
            ),
        ]

    evidence_refs = {e.api_ref for e in batch.evidence}
    tracked_namespaces = _tracked_namespaces_for_checks(batch)
    sources = _collect_generated_sources(handle)
    declared_misses: list[str] = []
    ast_misses: list[str] = []
    for path, source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # The compile-syntax check above flagged the error
            # separately; skip the AST diff for this file.
            continue
        declared = extract_declared_refs(tree)
        for ref in sorted(declared - evidence_refs):
            declared_misses.append(f"{path.name}:{ref}")
        ast_refs = extract_ast_refs(tree, namespaces=tracked_namespaces)
        for ref in sorted(ast_refs - evidence_refs):
            ast_misses.append(f"{path.name}:{ref}")

    return [
        CheckResult(
            name="capability_evidence_check[declared]",
            passed=not declared_misses,
            severity="info" if not declared_misses else "error",
            detail=(
                "; ".join(declared_misses)
                if declared_misses
                else "all __capability_evidence__ entries covered by evidence.yaml"
            ),
        ),
        CheckResult(
            name="capability_evidence_check[ast]",
            passed=not ast_misses,
            severity="info" if not ast_misses else "error",
            detail=(
                "; ".join(ast_misses)
                if ast_misses
                else "all tracked AST refs covered by evidence.yaml"
            ),
        ),
    ]


def _tracked_namespaces_for_checks(batch: CapabilityEvidenceBatch) -> tuple[str, ...]:
    tracked = set(batch.tracked_namespaces)
    tracked.update(hint.namespace for hint in batch.hints)
    for evidence in batch.evidence:
        namespace = evidence.namespace or evidence.package
        if namespace:
            tracked.add(namespace)
    return tuple(sorted(tracked))


def _collect_generated_sources(handle: PlanFolder) -> list[tuple[Path, str]]:
    """Read every generated task implementation + test module from disk."""
    out: list[tuple[Path, str]] = []
    for d, glob in (
        (handle.tasks_pkg_dir(), "*.py"),
        (handle.tests_dir(), "test_*.py"),
    ):
        if not d.exists():
            continue
        for path in sorted(d.glob(glob)):
            if path.name == "__init__.py":
                continue
            try:
                out.append((path, path.read_text(encoding="utf-8")))
            except OSError:
                continue
    return out


def _load_contract_for_validation(
    handle: PlanFolder,
    checks: list[CheckResult],
) -> WorkflowContract | None:
    ir_yaml_path = handle.ir_dir() / "workflow.yaml"
    try:
        contract_dict = default_compiler.yaml_to_ir(ir_yaml_path.read_text())
        contract = default_compiler.dict_to_contract(contract_dict)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        checks.append(
            CheckResult(
                name="workflow_ir_parseable",
                passed=False,
                severity="error",
                detail=str(exc),
            )
        )
        return None
    except Exception as exc:
        checks.append(
            CheckResult(
                name="workflow_ir_parseable",
                passed=False,
                severity="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return None
    checks.append(CheckResult(name="workflow_ir_parseable", passed=True, severity="info"))
    return contract


def _run_handoff_validation(
    handle: PlanFolder,
    handoff: PlanRunHandoff,
    *,
    contract: WorkflowContract | None = None,
    prefix: str = "",
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if contract is None:
        contract = _load_contract_for_validation(handle, checks)

    workflow = _load_handoff_workflow(handoff, checks, prefix=prefix)
    if workflow is None or contract is None:
        return checks

    report = validate_workflow_contract(contract, spec=workflow)
    if report.ok:
        checks.append(
            CheckResult(
                name=f"{prefix}workflow_contract_valid",
                passed=True,
                severity="info",
            )
        )
    for issue in report.issues:
        checks.append(
            CheckResult(
                name=f"{prefix}workflow_contract[{issue.check_id.value}:{issue.target}]",
                passed=False,
                severity=issue.severity,
                detail=issue.message + (f" Hint: {issue.hint}" if issue.hint else ""),
            )
        )
    return checks


def _load_handoff_workflow(
    handoff: PlanRunHandoff,
    checks: list[CheckResult],
    *,
    prefix: str = "",
) -> Workflow | None:
    source_root = handoff.experiment_workspace_path / handoff.source_root
    if not source_root.exists():
        checks.append(
            CheckResult(
                name=f"{prefix}handoff_source_root_exists",
                passed=False,
                severity="error",
                detail=f"missing {source_root}",
            )
        )
        return None
    checks.append(
        CheckResult(name=f"{prefix}handoff_source_root_exists", passed=True, severity="info")
    )

    try:
        module = _import_fresh_from_source_root(source_root, handoff.entrypoint_module)
        checks.append(
            CheckResult(
                name=f"{prefix}handoff_entrypoint_imports",
                passed=True,
                severity="info",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                name=f"{prefix}handoff_entrypoint_imports",
                passed=False,
                severity="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return None

    try:
        entrypoint = getattr(module, handoff.entrypoint_symbol)
    except AttributeError as exc:
        checks.append(
            CheckResult(
                name=f"{prefix}handoff_entrypoint_symbol_exists",
                passed=False,
                severity="error",
                detail=str(exc),
            )
        )
        return None
    checks.append(
        CheckResult(name=f"{prefix}handoff_entrypoint_symbol_exists", passed=True, severity="info")
    )

    try:
        candidate = entrypoint() if callable(entrypoint) else entrypoint
    except Exception as exc:
        checks.append(
            CheckResult(
                name=f"{prefix}handoff_entrypoint_returns_workflow",
                passed=False,
                severity="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return None
    is_workflow = isinstance(candidate, Workflow)
    checks.append(
        CheckResult(
            name=f"{prefix}handoff_entrypoint_returns_workflow",
            passed=is_workflow,
            severity="info" if is_workflow else "error",
            detail="" if is_workflow else f"got {type(candidate).__name__}",
        )
    )
    return candidate if is_workflow else None


def _import_fresh_from_source_root(source_root: Path, module_name: str) -> ModuleType:
    source_root_str = str(source_root)
    root_name = module_name.split(".", 1)[0]
    affected = {
        name: module
        for name, module in sys.modules.items()
        if name == root_name or name.startswith(root_name + ".")
    }
    for name in affected:
        sys.modules.pop(name, None)
    sys.path.insert(0, source_root_str)
    try:
        return importlib.import_module(module_name)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(source_root_str)
        for name in list(sys.modules):
            if name == root_name or name.startswith(root_name + "."):
                sys.modules.pop(name, None)
        sys.modules.update(affected)


def _checks_passed(checks: tuple[CheckResult, ...] | list[CheckResult]) -> bool:
    return not any(check.severity == "error" and not check.passed for check in checks)


def _summarize_checks(checks: tuple[CheckResult, ...] | list[CheckResult]) -> str:
    return (
        f"{sum(1 for c in checks if c.passed)} of {len(checks)} checks passed; "
        f"{sum(1 for c in checks if c.severity == 'error' and not c.passed)} errors, "
        f"{sum(1 for c in checks if c.severity == 'warning')} warnings."
    )


def _review_status(*, decision: ReviewDecision, validation_passed: bool) -> str:
    if validation_passed and decision.approved:
        return "approved"
    if validation_passed:
        return "ready_for_review"
    if decision.approved and decision.override_validation:
        return "approved_with_override"
    return "validation_failed"


def _final_status(
    *,
    decision: ReviewDecision,
    validation_passed: bool,
    ready_for_run: bool,
) -> str:
    if ready_for_run:
        return "ready_for_run"
    if decision.approved and decision.override_validation:
        return "approved_with_override"
    if not validation_passed:
        return "validation_failed"
    if decision.approved:
        return "approved"
    return "ready_for_review"


async def _read_yaml_into[T: BaseModel](path: Path, model: type[T], *, fallback: str) -> T:
    """Best-effort load — falls back to model defaults if the file is
    not a structured-payload mirror of ``model``.

    The on-disk markdown is a human-readable rendering, not a
    round-trippable representation. This helper instead reads any
    ``manifest.yaml`` companion section keyed by ``fallback``;
    if absent, returns a model with empty defaults so PlanReviewView
    can still be constructed for review."""
    del path
    # v1 keeps things simple: always return defaults. Sub-spec 07 will
    # extend manifest.yaml to carry the structured digest / plan brief
    # so the review view can show real values.
    if model is ReportDigest:
        return ReportDigest(
            summary="(see " + fallback + " on disk)",
            experimental_goal="(see workspace)",
        )  # type: ignore[return-value]
    if model is PlanBrief:
        return PlanBrief(
            overview="(see " + fallback + " on disk)",
            chosen_method="(see workspace)",
        )  # type: ignore[return-value]
    raise NotImplementedError(f"_read_yaml_into does not handle {model.__name__}")


def _load_manifest_from_disk(manifest_path: Path) -> PlanManifest | None:
    if not manifest_path.exists():
        return None
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        return None
    # Drop extension sections so PlanManifest's strict schema accepts it.
    raw_payload = {k: v for k, v in raw.items() if k not in {"handoff", "plan_mode"}}
    try:
        return PlanManifest.model_validate(raw_payload)
    except Exception:
        return None


def _build_manifest_stub(plan_id: str, workflow_ir_path: Path) -> PlanManifest:
    return PlanManifest(
        plan_id=plan_id,
        created_at=_utcnow(),
        report_source="report/original.md",
        workflow_ir_path=workflow_ir_path,
    )


def _persist_manifest_with_handoff(
    handle: PlanFolder,
    manifest: PlanManifest,
    handoff: PlanRunHandoff,
    *,
    decision: ReviewDecision,
    ready_for_run: bool,
    validation_passed: bool,
    validation_summary: str,
    status: str,
) -> None:
    """Write ``manifest.yaml`` with both the manifest fields and a
    nested ``handoff`` block plus a machine-readable ``plan_mode`` block.

    Sub-spec 03's ``write_manifest`` only knows about the manifest
    schema; PlanMode extension blocks are layered on top here.
    """
    payload = manifest.model_dump(mode="json")
    handoff_payload = json.loads(handoff.model_dump_json())
    payload["handoff"] = handoff_payload
    payload["plan_mode"] = {
        "status": status,
        "validation_passed": validation_passed,
        "ready_for_run": ready_for_run,
        "approved": decision.approved,
        "approval_reason": decision.reason,
        "override": decision.override_validation,
        "validation_summary": validation_summary,
        "generated_at": handoff.created_at.isoformat(),
        "handoff": {
            "source_root": str(handoff.source_root),
            "module": handoff.entrypoint_module,
            "symbol": handoff.entrypoint_symbol,
        },
    }
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    atomic_write_text(handle.manifest_path(), text)


def _utcnow() -> datetime:
    """Wrapper kept around for monkey-patching in tests."""
    return datetime.now(UTC)


# ── Helpers ────────────────────────────────────────────────────────────────


def _expect_input[T](inputs: object, key: str, expected: type[T]) -> T:
    """Pull and type-narrow one upstream from a multi-dep ``ctx.inputs``.

    The caller's class name is reconstructed from :func:`sys._getframe`
    so the error message points at the actual node — Phase 5 wires four
    distinct callers (``GenerateWorkflowSkeleton`` /
    ``GenerateTaskTests`` / ``GenerateTaskImplementations`` /
    ``ValidateWorkspace``) through this helper.
    """
    caller = _caller_class_name() or "Plan node"
    if not isinstance(inputs, dict):
        raise TypeError(f"{caller} expected dict-shaped ctx.inputs; got {type(inputs).__name__}")
    inputs_dict = cast("dict[str, object]", inputs)
    value = inputs_dict.get(key)
    if not isinstance(value, expected):
        raise TypeError(
            f"{caller} expected ctx.inputs[{key!r}] of "
            f"type {expected.__name__}; got {type(value).__name__}"
        )
    return value


def _caller_class_name() -> str | None:
    """Walk one frame up to fetch the caller's ``self.__class__.__name__``.

    Returns ``None`` when the caller is not a bound method (e.g. a
    free-function call from a test). The lookup is intentionally
    shallow — it inspects exactly one frame and tolerates absent
    ``self`` rather than scanning the stack, so it stays cheap on the
    happy path.
    """
    frame = sys._getframe(2)  # caller-of-caller
    self_obj = frame.f_locals.get("self")
    if self_obj is None:
        return None
    return type(self_obj).__name__


def _write_text(path: Path, content: str) -> None:
    """Atomic-write a string through the workspace's helper.

    Centralized so tasks never call ``Path.write_text`` directly — the
    AST guard ``ac-006`` enforces this rule.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content)


def _validate_python(source: str, path: str) -> None:
    """Syntax-check ``source`` via :func:`compile` (no execution).

    Raises :class:`SkeletonCompileError` with the original
    :class:`SyntaxError` chained as ``__cause__``.
    """
    try:
        compile(source, path, "exec")
    except SyntaxError as exc:
        raise SkeletonCompileError(
            f"generated source did not compile ({path}): {exc.msg} at line {exc.lineno}",
            path=path,
        ) from exc


def _render_digest_markdown(digest: ReportDigest) -> str:
    lines = [
        "# Report digest",
        "",
        digest.summary,
        "",
        "## Experimental goal",
        "",
        digest.experimental_goal,
    ]
    for heading, items in (
        ("Scientific assumptions", digest.scientific_assumptions),
        ("Systems and variables", digest.systems_and_variables),
        ("Expected outputs", digest.expected_outputs),
        ("Missing information", digest.missing_information),
    ):
        if not items:
            continue
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines) + "\n"


def _render_plan_brief_markdown(plan: PlanBrief) -> str:
    lines = [
        "# Implementation plan",
        "",
        plan.overview,
        "",
        "## Chosen method",
        "",
        plan.chosen_method,
    ]
    if plan.rationale:
        lines.extend(["", "## Rationale", "", plan.rationale])
    if plan.stages:
        lines.extend(["", "## Stages", ""])
        lines.extend(f"{i}. {stage}" for i, stage in enumerate(plan.stages, start=1))
    return "\n".join(lines) + "\n"


def _render_workflow_module(contract: WorkflowContract) -> str:
    """Render ``src/experiment/workflow.py`` from a workflow contract.

    Generates a ``WORKFLOW`` constant built via
    :class:`~molexp.workflow.WorkflowBuilder`, with one ``.add(<TaskClass>())``
    per ``task_io`` entry. Dependencies are inferred from each input's
    ``source`` field — distinct sources, in order, become the
    ``depends_on`` list.
    """
    import_lines: list[str] = []
    add_lines: list[str] = []
    for task_io in contract.task_io:
        cls_name = _camel_case(task_io.task_id)
        import_lines.append(f"from .tasks.{task_io.task_id} import {cls_name}")
        deps: list[str] = []
        seen: set[str] = set()
        for inp in task_io.inputs:
            if inp.source and inp.source not in seen:
                deps.append(inp.source)
                seen.add(inp.source)
        if deps:
            depends_on_repr = ", ".join(repr(d) for d in deps)
            add_lines.append(
                f"    .add({cls_name}(), name={task_io.task_id!r}, depends_on=[{depends_on_repr}])"
            )
        else:
            add_lines.append(f"    .add({cls_name}(), name={task_io.task_id!r})")

    imports = "\n".join(import_lines) if import_lines else "# (workflow contract has no tasks)"

    body = "\n".join(add_lines) if add_lines else "    # no tasks"
    workflow_name = contract.workflow_id or "experiment_workflow"

    return (
        '"""Generated experiment workflow.\n'
        "\n"
        "This module is regenerated by PlanMode's GenerateWorkflowSkeleton\n"
        "task. Edit the per-task modules under ``tasks/`` to fill in the\n"
        "implementations; do not hand-edit this file unless you intend to\n"
        "diverge from the plan.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from molexp.workflow import WorkflowBuilder\n"
        f"{imports}\n"
        "\n"
        "WORKFLOW = (\n"
        f"    WorkflowBuilder(name={workflow_name!r})\n"
        f"{body}\n"
        "    .build()\n"
        ")\n"
        "\n"
        "\n"
        "def create_workflow():\n"
        "    return WORKFLOW\n"
    )


def _camel_case(task_id: str) -> str:
    """Convert ``snake_case_task_id`` to ``CamelCaseClassName``."""
    parts = [p for p in task_id.replace("-", "_").split("_") if p]
    return "".join(p.capitalize() or "_" for p in parts) or "Task"


_PACKAGE_INIT_BODY = (
    '"""Generated experiment package — re-export the WORKFLOW constant."""\n'
    "\n"
    "from molexp.workspace import Workspace as _Workspace  # noqa: F401 — keeps the package importable in offline tests\n"
    "\n"
)

_TASKS_INIT_BODY = (
    '"""Generated experiment-task package.\n'
    "\n"
    "Per-task modules are written by PlanMode's GenerateTaskImplementations\n"
    "node (sub-spec 06). v1 of the skeleton lays down the package marker\n"
    'only — fill in concrete imports as the per-task files appear."""\n'
    "\n"
)
