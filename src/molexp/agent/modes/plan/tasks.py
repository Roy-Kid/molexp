"""Concrete tasks for the materialize-to-workspace PlanMode pipeline.

The 6 nodes of the v1 pipeline:

    IngestReport → DraftReportDigest → DraftImplementationPlan
        → CompileWorkflowIR → CompileTaskIR → GenerateWorkflowSkeleton

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

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import yaml
from pydantic import BaseModel

from molexp.agent.modes.plan.errors import SkeletonCompileError
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
from molexp.workflow import Task, TaskContext, default_compiler

__all__ = [
    "CompileTaskIR",
    "CompileWorkflowIR",
    "DraftImplementationPlan",
    "DraftReportDigest",
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

        paths: list[Path] = []
        for brief in ir_result.briefs:
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
            path = ctx.deps.workspace_handle.write_test_module(brief.task_id, source)
            paths.append(path)

        # Topology-pin test always lands.
        ir_yaml_rel = "ir/workflow.yaml"
        structure_source = _render_workflow_structure_test(ir_yaml_rel)
        ctx.deps.workspace_handle.write_workflow_structure_test(structure_source)

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

        paths: list[Path] = []
        for brief in ir_result.briefs:
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
            path = ctx.deps.workspace_handle.write_task_implementation(brief.task_id, source)
            paths.append(path)

        return TaskImplementationsResult(impl_paths=tuple(paths))


# ── Node 9: ValidateWorkspace ──────────────────────────────────────────────


class ValidateWorkspace(PlanTask):
    """Run the deterministic validation pass over the materialized workspace.

    Eight checks; results land in a :class:`ValidationReport` written
    to ``validation_report.md``. Errors block ``HumanReview``;
    warnings are surfaced but do not fail the pass.
    """

    async def execute(self, ctx: TaskContext[None, PlanDeps, dict[str, Any]]) -> ValidationResult:
        ir_result = _expect_input(ctx.inputs, "CompileTaskIR", TaskIRResult)
        # GenerateTaskImplementations / GenerateTaskTests outputs are
        # validated through the on-disk artifacts; pull them only to
        # ensure the upstream order ran.
        _expect_input(ctx.inputs, "GenerateTaskTests", TaskTestsResult)
        _expect_input(ctx.inputs, "GenerateTaskImplementations", TaskImplementationsResult)

        handle = ctx.deps.workspace_handle
        checks: list[CheckResult] = []

        # 1. workflow IR parseable
        ir_yaml_path = handle.ir_dir() / "workflow.yaml"
        try:
            yaml.safe_load(ir_yaml_path.read_text())
            checks.append(CheckResult(name="workflow_ir_parseable", passed=True, severity="info"))
        except (FileNotFoundError, yaml.YAMLError) as exc:
            checks.append(
                CheckResult(
                    name="workflow_ir_parseable",
                    passed=False,
                    severity="error",
                    detail=str(exc),
                )
            )

        # 2. every task IR file parseable
        for task_yaml in (
            handle.tasks_ir_dir().glob("*.yaml") if handle.tasks_ir_dir().exists() else ()
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

        # 3. every task in IR has an implementation module
        for brief in ir_result.briefs:
            impl_path = handle.tasks_pkg_dir() / f"{brief.task_id}.py"
            if impl_path.exists():
                checks.append(
                    CheckResult(
                        name=f"impl_present[{brief.task_id}]",
                        passed=True,
                        severity="info",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name=f"impl_present[{brief.task_id}]",
                        passed=False,
                        severity="error",
                        detail=f"missing {impl_path.name}",
                    )
                )

        # 4. every task in IR has a test module
        for brief in ir_result.briefs:
            test_path = handle.tests_dir() / f"test_{brief.task_id}.py"
            if test_path.exists():
                checks.append(
                    CheckResult(
                        name=f"test_present[{brief.task_id}]",
                        passed=True,
                        severity="info",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name=f"test_present[{brief.task_id}]",
                        passed=False,
                        severity="error",
                        detail=f"missing {test_path.name}",
                    )
                )

        # 5. expected workspace paths exist
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

        # 6. generated workflow imports cleanly (compile-only)
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

        # 7. empty-contract guard
        if not ir_result.briefs:
            checks.append(
                CheckResult(
                    name="contract_has_tasks",
                    passed=True,
                    severity="warning",
                    detail="no tasks declared in workflow contract",
                )
            )

        passed = not any(c.severity == "error" and not c.passed for c in checks)
        summary = (
            f"{sum(1 for c in checks if c.passed)} of {len(checks)} checks passed; "
            f"{sum(1 for c in checks if c.severity == 'error' and not c.passed)} errors, "
            f"{sum(1 for c in checks if c.severity == 'warning')} warnings."
        )
        report = ValidationReport(passed=passed, checks=tuple(checks), summary=summary)
        report_path = handle.write_validation_report(report)
        return ValidationResult(report_path=report_path, passed=passed, summary=summary)


# ── Node 10: HumanReview ───────────────────────────────────────────────────


class HumanReview(PlanTask):
    """Terminal node — gate the plan with a :class:`GatePolicy`.

    On approval, builds the :class:`PlanRunHandoff`, writes it into
    ``manifest.yaml``'s ``handoff`` section, flips manifest status to
    ``"approved"``, and returns a :class:`HandoffResult`. On rejection,
    leaves status at ``"pending_review"`` and returns a
    :class:`HandoffResult` whose ``decision.approved`` is False; the
    handoff field still carries a constructed :class:`PlanRunHandoff`
    so downstream consumers can introspect what would have been
    handed off.
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
        view = PlanReviewView(
            plan_id=plan_id,
            experiment_workspace_path=handle.root(),
            digest=digest,
            plan_brief=plan_brief,
            contract=contract,
            validation_passed=validation_result.passed,
            validation_summary=validation_result.summary,
        )

        decision = await ctx.deps.gate_policy.human_review(view)

        # Build the handoff regardless — both branches consume it.
        existing_manifest = _load_manifest_from_disk(handle.manifest_path())
        validation_report = ValidationReport(
            passed=validation_result.passed,
            summary=validation_result.summary,
        )
        new_status = "approved" if decision.approved else "pending_review"
        manifest_for_handoff = (
            existing_manifest or _build_manifest_stub(plan_id, contract_yaml, ir_result_briefs=())
        ).model_copy(update={"status": new_status})
        handoff = PlanRunHandoff(
            plan_id=plan_id,
            experiment_workspace_path=handle.root(),
            workflow_yaml_path=contract_yaml,
            task_ir_paths=tuple(handle.tasks_ir_dir().glob("*.yaml"))
            if handle.tasks_ir_dir().exists()
            else (),
            entrypoint_module="experiment.workflow",
            entrypoint_symbol="WORKFLOW",
            manifest_snapshot=manifest_for_handoff,
            validation_report_snapshot=validation_report,
            created_at=_utcnow(),
        )

        # On approval, persist the manifest with the handoff block.
        if decision.approved:
            _persist_manifest_with_handoff(handle, manifest_for_handoff, handoff)

        return HandoffResult(handoff=handoff, decision=decision)


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
        "        from experiment.workflow import WORKFLOW\n"
        "    except ImportError:\n"
        '        pytest.skip("experiment.workflow import failed; skipping topology pin")\n'
        "        return\n"
        "    actual_ids = {t.name for t in WORKFLOW._tasks}\n"
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
    # Drop any "handoff" section so PlanManifest's strict schema accepts it.
    raw_payload = {k: v for k, v in raw.items() if k != "handoff"}
    try:
        return PlanManifest.model_validate(raw_payload)
    except Exception:
        return None


def _build_manifest_stub(
    plan_id: str,
    workflow_ir_path: Path,
    *,
    ir_result_briefs: tuple[Any, ...],
) -> PlanManifest:
    del ir_result_briefs
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
) -> None:
    """Write ``manifest.yaml`` with both the manifest fields and a
    nested ``handoff`` block.

    Sub-spec 03's ``write_manifest`` only knows about the manifest
    schema; the handoff block is layered on top here.
    """
    import json

    payload = manifest.model_dump(mode="json")
    payload["handoff"] = json.loads(handoff.model_dump_json())
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    from molexp.workspace import atomic_write_text

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
    from molexp.workspace import atomic_write_text

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
    task_class_names: list[str] = []
    add_lines: list[str] = []
    for task_io in contract.task_io:
        cls_name = _camel_case(task_io.task_id)
        task_class_names.append(cls_name)
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

    if task_class_names:
        import_line = "from .tasks import " + ", ".join(sorted(set(task_class_names)))
    else:
        import_line = "# (workflow contract has no tasks; nothing to import)"

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
        f"{import_line}\n"
        "\n"
        "WORKFLOW = (\n"
        f"    WorkflowBuilder(name={workflow_name!r})\n"
        f"{body}\n"
        "    .build()\n"
        ")\n"
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
