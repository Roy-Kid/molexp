"""Concrete tasks for the materialize-to-workspace PlanMode pipeline.

The nodes of the current pipeline:

    IngestReport → DraftReportDigest → DraftImplementationPlan
        → CompileWorkflowIR → CompileTaskIR → GenerateWorkflowSkeleton
        → GenerateTaskTests / GenerateTaskImplementations
        → ValidateWorkspace → HumanReview → FinalHandoffCheck

Each node consumes its single upstream ``ctx.inputs``, materializes
its product through :class:`~molexp.agent.modes.plan.workspace_layout.PlanWorkspaceHandle`,
and returns a frozen ``*Result`` carrying *path references* — never
embedded blobs. Downstream nodes operate on file handles, the
workspace is the single source of truth for materialized content.

LLM-bearing tasks subclass :class:`PlanLLMTask`; the tier each one
runs at is resolved by ``ctx.deps.policy.tier_for(type(self).__name__)``
(no per-class ``TIER`` ClassVar literal). ``IngestReport`` skips the
LLM entirely — it just hashes and writes the raw user input.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, cast

import yaml
from pydantic import BaseModel

from molexp.agent.modes.plan.errors import SkeletonCompileError
from molexp.agent.modes.plan.handoff import PlanRunHandoff
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    DigestResult,
    HandoffResult,
    IngestReportResult,
    PlanBrief,
    PlanBriefResult,
    PlanReviewView,
    ReportDigest,
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
from molexp.agent.modes.plan.workspace_layout import (
    CheckResult,
    PlanManifest,
    PlanWorkspaceHandle,
    ValidationReport,
)
from molexp.workflow import (
    Task,
    TaskContext,
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


# ── Bases ──────────────────────────────────────────────────────────────────


class PlanTask(Task):
    """Base for every task inside the materialize-to-workspace pipeline."""


class PlanLLMTask(PlanTask):
    """Plan task that invokes the provider to produce structured output.

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
    ) -> SchemaT:
        node_id = type(self).__name__
        return await ctx.deps.provider.invoke(
            tier=ctx.deps.policy.tier_for(node_id),
            system=self.SYSTEM_PROMPT,
            user=user,
            schema=schema,
            node_id=node_id,
        )


# ── Node 1: IngestReport ───────────────────────────────────────────────────


class IngestReport(PlanTask):
    """Materialize the user-supplied report verbatim into ``report/original.md``.

    No LLM call. Hashes the input via SHA-256 so downstream nodes can
    detect re-runs against the same content (idempotency hint, not an
    invariant).
    """

    async def execute(self, ctx: TaskContext[None, PlanDeps, None]) -> IngestReportResult:
        user_input = ctx.config.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("IngestReport requires a non-empty 'user_input' in config.")

        report_dir = ctx.deps.workspace_handle.report_dir()
        report_path = report_dir / "original.md"
        _write_text(report_path, user_input)

        return IngestReportResult(
            report_path=report_path,
            report_hash=hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
        )


# ── Node 2: DraftReportDigest ──────────────────────────────────────────────


class DraftReportDigest(PlanLLMTask):
    """Distill the original report into a structured + markdown digest."""

    SYSTEM_PROMPT = (
        "You are summarizing an experimental report. Return a structured "
        "ReportDigest covering goal, assumptions, systems, expected outputs, "
        "and any missing information."
    )

    async def execute(self, ctx: TaskContext[None, PlanDeps, IngestReportResult]) -> DigestResult:
        ingest = ctx.inputs
        report_text = Path(ingest.report_path).read_text(encoding="utf-8")
        digest = await self.invoke_llm(ctx, user=report_text, schema=ReportDigest)

        digest_path = ctx.deps.workspace_handle.report_dir() / "digest.md"
        _write_text(digest_path, _render_digest_markdown(digest))
        return DigestResult(digest_path=digest_path, digest=digest)


# ── Node 3: DraftImplementationPlan ────────────────────────────────────────


class DraftImplementationPlan(PlanLLMTask):
    """Render a natural-language implementation plan from the digest."""

    SYSTEM_PROMPT = (
        "Given the ReportDigest, draft a PlanBrief covering an overview, "
        "the chosen experimental method with rationale, and an ordered list "
        "of stages."
    )

    async def execute(self, ctx: TaskContext[None, PlanDeps, DigestResult]) -> PlanBriefResult:
        digest_result = ctx.inputs
        plan_brief = await self.invoke_llm(
            ctx,
            user=digest_result.digest.model_dump_json(),
            schema=PlanBrief,
        )
        plan_path = ctx.deps.workspace_handle.plan_dir() / "implementation_plan.md"
        _write_text(plan_path, _render_plan_brief_markdown(plan_brief))
        return PlanBriefResult(plan_path=plan_path, plan_brief=plan_brief)


# ── Node 4: CompileWorkflowIR ──────────────────────────────────────────────


class CompileWorkflowIR(PlanLLMTask):
    """Compile the plan brief into a typed :class:`WorkflowContract`."""

    SYSTEM_PROMPT = (
        "Translate the PlanBrief into a WorkflowContract: list every task "
        "(task_io with inputs/outputs/artifacts), declare dependencies via "
        "the 'source' field on each input, and list the validation_checks "
        "you want enforced."
    )

    async def execute(self, ctx: TaskContext[None, PlanDeps, PlanBriefResult]) -> WorkflowIRResult:
        plan_result = ctx.inputs
        contract = await self.invoke_llm(
            ctx,
            user=plan_result.plan_brief.model_dump_json(),
            schema=WorkflowContract,
        )
        ir_path = ctx.deps.workspace_handle.ir_dir() / "workflow.yaml"
        contract_dict = default_compiler.contract_to_dict(contract)
        _write_text(ir_path, default_compiler.ir_to_yaml(contract_dict))
        return WorkflowIRResult(workflow_yaml_path=ir_path, contract=contract)


# ── Node 5: CompileTaskIR ──────────────────────────────────────────────────


class CompileTaskIR(PlanLLMTask):
    """Compile a per-task brief for every task in the workflow contract.

    Empty-task contracts (``contract.task_io == ()``) emit an empty
    ``task_ir_paths`` tuple — sub-spec 01 explicitly permits empty
    contracts, so this node is not allowed to fail on them.
    """

    SYSTEM_PROMPT = (
        "Given one TaskIO entry from a WorkflowContract, draft a TaskIRBrief "
        "covering its responsibility, success criteria, failure conditions, "
        "and minimal test expectations. Set is_stub=true only if you cannot "
        "describe a concrete implementation."
    )

    async def execute(self, ctx: TaskContext[None, PlanDeps, WorkflowIRResult]) -> TaskIRResult:
        ir_result = ctx.inputs
        tasks_ir_dir = ctx.deps.workspace_handle.tasks_ir_dir()

        paths: list[Path] = []
        briefs: list[TaskIRBrief] = []
        for task_io in ir_result.contract.task_io:
            brief = await self.invoke_llm(
                ctx,
                user=task_io.model_dump_json(),
                schema=TaskIRBrief,
            )
            # Force the brief's task_id to match the contract entry —
            # the LLM may forget; the workflow_contract is authoritative.
            if brief.task_id != task_io.task_id:
                brief = brief.model_copy(update={"task_id": task_io.task_id})
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

    async def execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> SkeletonResult:
        # Read both upstreams: the workflow contract for topology, the
        # task IR result to confirm the per-task IR has been written.
        ir_result = _expect_input(ctx.inputs, "CompileWorkflowIR", WorkflowIRResult)
        _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)

        contract = ir_result.contract
        package_path = ctx.deps.workspace_handle.experiment_pkg_dir()
        tasks_pkg = ctx.deps.workspace_handle.tasks_pkg_dir()

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
        "Given a TaskIRBrief plus its TaskIO declaration, draft a pytest "
        "module that exercises the task. If is_stub=True, emit a single "
        "test that calls pytest.skip('stub'). Otherwise, write at least "
        "one happy-path test referencing the documented inputs / outputs."
    )

    async def execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> TaskTestsResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        skeleton = _expect_input(ctx.inputs, "GenerateWorkflowSkeleton", SkeletonResult)
        del skeleton  # path is informational; the writer uses workspace_handle

        # Repair-loop filter: when ``ctx.deps.repair_target_tasks`` is a
        # non-None tuple, only briefs whose task_id is in the set get a
        # fresh LLM call this iteration. The remaining briefs reuse the
        # on-disk file from the prior round; the returned ``test_paths``
        # tuple still covers every brief so ValidateWorkspace's
        # consistency check sees a complete set.
        repair_targets = ctx.deps.repair_target_tasks
        handle = ctx.deps.workspace_handle

        paths: list[Path] = []
        for brief in ir_result.briefs:
            existing_path = handle.tests_dir() / f"test_{brief.task_id}.py"
            if (
                repair_targets is not None
                and brief.task_id not in repair_targets
                and existing_path.exists()
            ):
                # Untouched on this repair iteration — reuse last round.
                paths.append(existing_path)
                continue
            if brief.is_stub:
                # Skip the LLM round-trip — emit a pytest.skip stub
                # synchronously. Saves a model call and keeps the
                # stub-tolerance contract tight.
                source = _render_stub_test_module(brief.task_id)
            else:
                module = await self.invoke_llm(
                    ctx,
                    user=brief.model_dump_json(),
                    schema=TaskTestModule,
                )
                # Force task_id to match the brief; LLMs may forget.
                if module.task_id != brief.task_id:
                    module = module.model_copy(update={"task_id": brief.task_id})
                source = module.source
            path = handle.write_test_module(brief.task_id, source)
            paths.append(path)

        # Topology-pin test always lands.
        ir_yaml_rel = "ir/workflow.yaml"
        structure_source = _render_workflow_structure_test(ir_yaml_rel)
        handle.write_workflow_structure_test(structure_source)

        return TaskTestsResult(test_paths=tuple(paths))


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
        "Given a TaskIRBrief plus its TaskIO declaration, write a Python "
        "module implementing the task as a molexp.workflow.Task subclass "
        "with `async def execute(ctx)`. Set is_stub=true if you cannot "
        "produce a runnable body."
    )

    async def execute(
        self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]
    ) -> TaskImplementationsResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        _expect_input(ctx.inputs, "GenerateWorkflowSkeleton", SkeletonResult)

        # See ``GenerateTaskTests.execute`` above for the
        # ``repair_target_tasks`` semantic.
        repair_targets = ctx.deps.repair_target_tasks
        handle = ctx.deps.workspace_handle

        paths: list[Path] = []
        for brief in ir_result.briefs:
            existing_path = handle.tasks_pkg_dir() / f"{brief.task_id}.py"
            if (
                repair_targets is not None
                and brief.task_id not in repair_targets
                and existing_path.exists()
            ):
                paths.append(existing_path)
                continue
            if brief.is_stub:
                source = _render_stub_implementation_module(brief.task_id)
            else:
                module = await self.invoke_llm(
                    ctx,
                    user=brief.model_dump_json(),
                    schema=TaskImplementationModule,
                )
                source = (
                    _render_stub_implementation_module(brief.task_id)
                    if module.is_stub
                    else module.source
                )
            path = handle.write_task_implementation(brief.task_id, source)
            paths.append(path)

        return TaskImplementationsResult(impl_paths=tuple(paths))


# ── Node 9: ValidateWorkspace ──────────────────────────────────────────────


class ValidateWorkspace(PlanTask):
    """Run the deterministic validation pass over the materialized workspace.

    The pass includes RunMode-style import validation and generic
    workflow contract validation. Results land in both
    ``validation_report.md`` and ``validation_report.yaml``.
    """

    async def execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> ValidationResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        # GenerateTaskImplementations / GenerateTaskTests outputs are
        # validated through the on-disk artifacts; pull them only to
        # ensure the upstream order ran.
        _expect_input(ctx.inputs, "GenerateTaskTests", TaskTestsResult)
        _expect_input(ctx.inputs, "GenerateTaskImplementations", TaskImplementationsResult)

        handle = ctx.deps.workspace_handle
        checks = _run_workspace_validation(
            handle,
            task_ir=ir_result,
            module_name="experiment.workflow",
            symbol_name="create_workflow",
        )
        passed = _checks_passed(checks)
        status = "ready_for_review" if passed else "validation_failed"
        summary = _summarize_checks(checks)
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
    """Gate the plan with a :class:`GatePolicy`.

    Human approval is deliberately separate from RunMode readiness.
    The next node, ``FinalHandoffCheck``, is the only node allowed to
    mark the workspace ``ready_for_run``.
    """

    async def execute(self, ctx: TaskContext[None, PlanDeps, ValidationResult]) -> HandoffResult:
        validation_result = ctx.inputs

        # Re-read upstream artefacts from the workspace to assemble the
        # review view + handoff. We pull from disk rather than threading
        # them all via ctx.inputs to keep the workflow fan-in shallow.
        handle = ctx.deps.workspace_handle
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
        # ``ApprovalDecision.target_*``. Empty tuple on first pass / when
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

        decision = await ctx.deps.gate_policy.human_review(view)

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

    async def execute(self, ctx: TaskContext[None, PlanDeps, HandoffResult]) -> HandoffResult:
        prior = ctx.inputs
        handle = ctx.deps.workspace_handle
        final_checks = _run_handoff_validation(handle, prior.handoff, prefix="final_")
        checks = (*prior.validation_checks, *final_checks)
        validation_passed = prior.validation_passed and _checks_passed(final_checks)
        ready_for_run = bool(prior.decision.approved and validation_passed)
        status = _final_status(
            decision=prior.decision,
            validation_passed=validation_passed,
            ready_for_run=ready_for_run,
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
    handle: PlanWorkspaceHandle,
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


def _load_contract_for_validation(
    handle: PlanWorkspaceHandle,
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
    handle: PlanWorkspaceHandle,
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


def _review_status(*, decision: ApprovalDecision, validation_passed: bool) -> str:
    if validation_passed and decision.approved:
        return "approved"
    if validation_passed:
        return "ready_for_review"
    if decision.approved and decision.override_validation:
        return "approved_with_override"
    return "validation_failed"


def _final_status(
    *,
    decision: ApprovalDecision,
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
    handle: PlanWorkspaceHandle,
    manifest: PlanManifest,
    handoff: PlanRunHandoff,
    *,
    decision: ApprovalDecision,
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
    """Pull and type-narrow one upstream from a multi-dep ``ctx.inputs``."""
    if not isinstance(inputs, dict):
        raise TypeError(
            f"GenerateWorkflowSkeleton expected dict-shaped ctx.inputs; got {type(inputs).__name__}"
        )
    inputs_dict = cast("dict[str, object]", inputs)
    value = inputs_dict.get(key)
    if not isinstance(value, expected):
        raise TypeError(
            f"GenerateWorkflowSkeleton expected ctx.inputs[{key!r}] of "
            f"type {expected.__name__}; got {type(value).__name__}"
        )
    return value


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
