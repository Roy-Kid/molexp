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
from pathlib import Path
from typing import Any, ClassVar, cast

import yaml
from pydantic import BaseModel

from molexp.agent.modes.plan.errors import SkeletonCompileError
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    IngestReportResult,
    PlanBrief,
    PlanBriefResult,
    ReportDigest,
    SkeletonResult,
    TaskIRBrief,
    TaskIRResult,
    WorkflowContract,
    WorkflowIRResult,
)
from molexp.workflow import Task, TaskContext, default_compiler

__all__ = [
    "CompileTaskIR",
    "CompileWorkflowIR",
    "DraftImplementationPlan",
    "DraftReportDigest",
    "GenerateWorkflowSkeleton",
    "IngestReport",
    "PlanLLMTask",
    "PlanTask",
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
