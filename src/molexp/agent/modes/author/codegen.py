"""AuthorMode codegen stages — recreated against the typed plan surface.

The codegen nodes the old ``plan/tasks.py`` owned (``CompileWorkflowIR``,
``CompileTaskIR``, ``GenerateWorkflowSkeleton``, ``GenerateTaskTests``,
``GenerateTaskImplementations``, ``ValidateWorkspace``,
``FinalHandoffCheck``) are recreated here as **plain async functions** —
AuthorMode's own pipeline is a plain async stage sequence on the harness,
not a ``pydantic_graph`` graph. Each function reads the typed
:class:`~molexp.agent.modes._planning.PlanGraph` /
:class:`~molexp.agent.modes._planning.CapabilityGraph` plus the lowered
:class:`~molexp.workflow.WorkflowContract`; the un-evidenced-symbol gate
(:func:`~molexp.agent.modes.author.codegen_evidence.validate_codegen_evidence`)
keys off ``CapabilityGraph`` nodes.

The LLM-bearing functions (``compile_task_ir``, ``generate_task_tests``,
``generate_task_implementations``) go through the
:class:`~molexp.agent.router.Router` structured path; the structural
renderers (``generate_workflow_skeleton``) are templated.
"""

from __future__ import annotations

import asyncio

import yaml
from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import CapabilityGraph, PlanGraph, PlanStep
from molexp.agent.modes.author.codegen_evidence import (
    MissingCapability,
    validate_codegen_evidence,
)
from molexp.agent.modes.author.renderers import (
    PACKAGE_INIT_BODY,
    TASKS_INIT_BODY,
    render_stub_implementation,
    render_stub_test,
    render_workflow_module,
    render_workflow_structure_test,
    validate_python,
)
from molexp.agent.modes.author.workspace_layout import MaterializedLayout
from molexp.agent.router import ModelTier, Router
from molexp.workflow import (
    TaskIO,
    WorkflowContract,
    default_compiler,
    validate_workflow_contract,
)

__all__ = [
    "CodegenError",
    "GeneratedModule",
    "TaskIRBrief",
    "WorkspaceValidation",
    "compile_task_ir",
    "generate_task_implementations",
    "generate_task_tests",
    "generate_workflow_skeleton",
    "validate_workspace",
    "write_workflow_ir",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class CodegenError(RuntimeError):
    """Raised when a codegen stage produces output that fails its gate.

    Carries the structured failure cause: a syntax error in generated
    source, or a tuple of un-evidenced :class:`MissingCapability` rows.
    """

    def __init__(self, message: str, *, missing: tuple[MissingCapability, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


# ── Structured codegen schemas ───────────────────────────────────────────


class TaskIRBrief(BaseModel):
    """The per-task IR brief the LLM drafts for one workflow task.

    Attributes:
        task_id: The task's stable id (must match the contract entry).
        responsibility: One sentence on what the task does.
        success_criteria: What a successful run produces.
        test_expectations: What the generated test should assert.
        is_stub: True when no capability evidence supports the task.
    """

    model_config = _FROZEN

    task_id: str
    responsibility: str
    success_criteria: tuple[str, ...] = ()
    test_expectations: tuple[str, ...] = ()
    is_stub: bool = False


class GeneratedModule(BaseModel):
    """One LLM-generated Python module (implementation or test).

    Attributes:
        task_id: The task this module belongs to.
        source: The full module source text.
        is_stub: True when the module is a no-op stub.
    """

    model_config = _FROZEN

    task_id: str
    source: str
    is_stub: bool = False


class WorkspaceValidation(BaseModel):
    """The verdict of :func:`validate_workspace`.

    Attributes:
        passed: Whether every check passed.
        summary: A human-readable one-line summary.
        failed_checks: Names of the failing checks.
    """

    model_config = _FROZEN

    passed: bool
    summary: str
    failed_checks: tuple[str, ...] = ()


# ── Stage: write the workflow IR ─────────────────────────────────────────


def write_workflow_ir(contract: WorkflowContract, *, layout: MaterializedLayout) -> str:
    """Serialize ``contract`` to ``ir/workflow.yaml`` and return the YAML text."""
    contract_dict = default_compiler.contract_to_dict(contract)
    yaml_text = default_compiler.ir_to_yaml(contract_dict)
    layout.write(layout.workflow_yaml_path(), yaml_text)
    return yaml_text


# ── Stage: compile per-task IR ───────────────────────────────────────────


_TASK_IR_SYSTEM_PROMPT = (
    "Given one TaskIO entry from a workflow contract plus the typed "
    "capability graph, draft a TaskIRBrief: the task's responsibility, "
    "success criteria, and minimal test expectations. Name only "
    "evidenced symbols drawn from the capability graph's api_refs. Set "
    "is_stub=true when no capability node supports the task — never "
    "invent a symbol to fill the gap."
)


async def compile_task_ir(
    *,
    router: Router,
    contract: WorkflowContract,
    capability_graph: CapabilityGraph,
    layout: MaterializedLayout,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[TaskIRBrief, ...]:
    """Draft a per-task IR brief for every task in ``contract``.

    Each brief is written to ``ir/tasks/<task_id>.yaml``. An empty
    contract yields an empty tuple (empty contracts are permitted).
    """
    if not contract.task_io:
        return ()

    async def _draft(tio: TaskIO) -> TaskIRBrief:
        user = (
            f"TaskIO:\n{tio.model_dump_json(indent=2)}\n\n"
            f"CapabilityGraph:\n{capability_graph.model_dump_json(indent=2)}"
        )
        brief = await router.complete_structured(
            tier=tier,
            system=_TASK_IR_SYSTEM_PROMPT,
            user=user,
            schema=TaskIRBrief,
            node_id=f"CompileTaskIR/{tio.task_id}",
        )
        if brief.task_id != tio.task_id:
            brief = brief.model_copy(update={"task_id": tio.task_id})
        return brief

    briefs = tuple(await asyncio.gather(*[_draft(tio) for tio in contract.task_io]))
    for brief in briefs:
        layout.write(
            layout.task_ir_path(brief.task_id),
            yaml.safe_dump(brief.model_dump(mode="json"), sort_keys=False),
        )
    return briefs


# ── Stage: generate the package skeleton ─────────────────────────────────


def generate_workflow_skeleton(contract: WorkflowContract, *, layout: MaterializedLayout) -> str:
    """Emit the templated experiment package skeleton; return workflow.py path.

    Writes ``src/experiment/__init__.py``, ``src/experiment/workflow.py``,
    and ``src/experiment/tasks/__init__.py``; every file is syntax-checked.

    Raises:
        CodegenError: when a generated file does not compile.
    """
    package_init = layout.package_dir() / "__init__.py"
    workflow_py = layout.workflow_py_path()
    tasks_init = layout.tasks_pkg_dir() / "__init__.py"
    workflow_source = render_workflow_module(contract)
    for source, path in (
        (PACKAGE_INIT_BODY, package_init),
        (workflow_source, workflow_py),
        (TASKS_INIT_BODY, tasks_init),
    ):
        try:
            validate_python(source, str(path))
        except SyntaxError as exc:
            raise CodegenError(f"generated skeleton did not compile ({path}): {exc.msg}") from exc
        layout.write(path, source)
    return str(workflow_py)


# ── Stage: generate per-task tests ───────────────────────────────────────


_TASK_TEST_SYSTEM_PROMPT = (
    "Given a TaskIRBrief plus its TaskIO declaration, draft a pytest "
    "module that exercises the task IN ISOLATION. The test MUST "
    "construct its own minimal synthetic inputs — it must never import, "
    "call, or consume the real output of an upstream task. Honour the "
    "brief's test_expectations: a 'synthetic input:' line names an "
    "input the test must build itself; an 'assert:' line names what it "
    "must check. Import only stdlib plus symbols evidenced in the "
    "capability graph. If is_stub=true, emit a single test calling "
    "pytest.skip('stub'). Return the full module source."
)


def _brief_with_test_sketch(brief: TaskIRBrief, step: PlanStep | None) -> TaskIRBrief:
    """Fold a plan step's ``IsolatedTestSketch`` into the brief.

    The step's ``test_sketch`` carries the minimal synthetic inputs the
    isolated test must build for itself and what it should assert; these
    are threaded into ``TaskIRBrief.test_expectations`` (prefixed so the
    test-generation prompt can tell them apart) rather than into a new
    schema. ``step is None`` — no matching plan step — leaves the brief
    unchanged.
    """
    if step is None:
        return brief
    sketch = step.test_sketch
    extra = tuple(f"synthetic input: {item}" for item in sketch.synthetic_inputs) + tuple(
        f"assert: {item}" for item in sketch.assertion_sketch
    )
    if not extra:
        return brief
    return brief.model_copy(update={"test_expectations": brief.test_expectations + extra})


async def generate_task_tests(
    *,
    router: Router,
    briefs: tuple[TaskIRBrief, ...],
    contract: WorkflowContract,
    capability_graph: CapabilityGraph,
    layout: MaterializedLayout,
    plan_graph: PlanGraph | None = None,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[str, ...]:
    """Generate one pytest module per task plus the topology-pin test.

    When ``plan_graph`` is supplied, each brief is augmented with the
    matching :class:`~molexp.agent.modes._planning.PlanStep`'s
    ``test_sketch`` — the isolated test's minimal synthetic inputs and
    assertion sketch — before generation, so the test exercises the task
    in isolation rather than on real upstream output.

    Returns the tuple of written test-file paths.
    """
    io_by_id = {tio.task_id: tio for tio in contract.task_io}
    paths: list[str] = []
    for brief in briefs:
        if plan_graph is not None:
            brief = _brief_with_test_sketch(brief, plan_graph.step_by_id(brief.task_id))
        if brief.is_stub:
            source = render_stub_test(brief.task_id)
        else:
            source = await _generate_one(
                router=router,
                brief=brief,
                tio=io_by_id.get(brief.task_id),
                capability_graph=capability_graph,
                system=_TASK_TEST_SYSTEM_PROMPT,
                node_prefix="GenerateTaskTests",
                tier=tier,
            )
        paths.append(str(layout.write(layout.task_test_path(brief.task_id), source)))

    structure_test = render_workflow_structure_test("ir/workflow.yaml")
    paths.append(
        str(layout.write(layout.tests_dir() / "test_workflow_structure.py", structure_test))
    )
    return tuple(paths)


# ── Stage: generate per-task implementations ─────────────────────────────


_TASK_IMPL_SYSTEM_PROMPT = (
    "Given a TaskIRBrief plus its TaskIO declaration, write a Python "
    "module implementing the task as a molexp.workflow.Task subclass "
    "with `async def execute(ctx)`. Call only symbols evidenced in the "
    "capability graph. If is_stub=true, the module must raise "
    "NotImplementedError. Return the full module source."
)


async def generate_task_implementations(
    *,
    router: Router,
    briefs: tuple[TaskIRBrief, ...],
    contract: WorkflowContract,
    capability_graph: CapabilityGraph,
    layout: MaterializedLayout,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[str, ...]:
    """Generate one runnable module per task. Returns the written paths."""
    io_by_id = {tio.task_id: tio for tio in contract.task_io}
    paths: list[str] = []
    for brief in briefs:
        if brief.is_stub:
            source = render_stub_implementation(brief.task_id)
        else:
            source = await _generate_one(
                router=router,
                brief=brief,
                tio=io_by_id.get(brief.task_id),
                capability_graph=capability_graph,
                system=_TASK_IMPL_SYSTEM_PROMPT,
                node_prefix="GenerateTaskImplementations",
                tier=tier,
            )
        paths.append(str(layout.write(layout.task_impl_path(brief.task_id), source)))
    return tuple(paths)


async def _generate_one(
    *,
    router: Router,
    brief: TaskIRBrief,
    tio: TaskIO | None,
    capability_graph: CapabilityGraph,
    system: str,
    node_prefix: str,
    tier: ModelTier,
) -> str:
    """Generate, syntax-check, and evidence-gate one module's source."""
    tio_json = tio.model_dump_json(indent=2) if tio is not None else "{}"
    user = (
        f"TaskIRBrief:\n{brief.model_dump_json(indent=2)}\n\n"
        f"TaskIO:\n{tio_json}\n\n"
        f"CapabilityGraph:\n{capability_graph.model_dump_json(indent=2)}"
    )
    module = await router.complete_structured(
        tier=tier,
        system=system,
        user=user,
        schema=GeneratedModule,
        node_id=f"{node_prefix}/{brief.task_id}",
    )
    where = f"{node_prefix}/{brief.task_id}"
    try:
        validate_python(module.source, where)
    except SyntaxError as exc:
        raise CodegenError(f"generated module did not compile ({where}): {exc.msg}") from exc
    missing = validate_codegen_evidence(module.source, capability_graph)
    if missing:
        refs = ", ".join(m.ref for m in missing)
        raise CodegenError(
            f"{where}: generated module references un-evidenced symbols: {refs}",
            missing=missing,
        )
    return module.source


# ── Stage: validate the materialized workspace ───────────────────────────


def validate_workspace(contract: WorkflowContract) -> WorkspaceValidation:
    """Run :func:`~molexp.workflow.validate_workflow_contract` over the IR.

    The materialized IR is the source of truth; this re-runs the
    workflow-layer contract checks and folds the report into a
    :class:`WorkspaceValidation` verdict.
    """
    report = validate_workflow_contract(contract)
    failed = tuple(issue.check_id.value for issue in report.issues if issue.severity == "error")
    if report.ok:
        summary = f"workspace validation passed ({len(contract.task_io)} task(s))"
    else:
        summary = f"workspace validation failed: {len(failed)} error(s)"
    return WorkspaceValidation(passed=report.ok, summary=summary, failed_checks=failed)
