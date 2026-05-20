"""The AuthorMode materialization pipeline — plain async stages.

:func:`materialize_plan` runs AuthorMode's codegen + debug + validation
work as a plain async sequence of harness stages — one ``async with
harness.stage(name): ...`` per stage. This is *not* a ``pydantic_graph``
graph: AuthorMode's own pipeline is plain async stages (the runnable
:class:`~molexp.workflow.Workflow` it *produces* is a separate concern,
built through the public ``molexp.workflow`` API).

Stages, in order:

1. ``LowerPlanGraph`` — lower the typed ``PlanGraph`` into a
   ``WorkflowContract`` and write ``ir/workflow.yaml``.
2. ``CompileTaskIR`` — draft one per-task IR brief per task.
3. ``GenerateWorkflowSkeleton`` — emit the experiment package skeleton.
4. ``GenerateTaskTests`` / ``GenerateTaskImplementations`` — per-task
   tests + implementations.
5. ``RunTaskDebugLoop`` — run each generated test in an isolated
   subprocess, repairing on failure.
6. ``ValidateWorkspace`` — re-validate the materialized IR.
7. ``WriteManifest`` — write ``manifest.yaml``.

All artefact paths are accumulated for the caller to emit
``artifact_written`` events.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.harness import AgentHarness
from molexp.agent.modes._planning import PlanDiff, PlanGraph
from molexp.agent.modes.author.codegen import (
    CodegenError,
    compile_task_ir,
    generate_task_implementations,
    generate_task_tests,
    generate_workflow_skeleton,
    validate_workspace,
    write_workflow_ir,
)
from molexp.agent.modes.author.debug_loop import run_task_debug_loop
from molexp.agent.modes.author.lowering import lower_plan_graph
from molexp.agent.modes.author.workspace_layout import MaterializedLayout
from molexp.workflow import ValidationReport

_LOG = get_logger(__name__)

__all__ = ["MaterializationOutcome", "materialize_plan"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_ENTRYPOINT_MODULE = "experiment.workflow"
_ENTRYPOINT_SYMBOL = "create_workflow"


class MaterializationOutcome(BaseModel):
    """The result of one :func:`materialize_plan` call.

    Attributes:
        plan_graph: The plan with ``compiled_contract_ref`` set.
        codegen_ok: Whether codegen + the debug loop fully succeeded.
        validation_report: The final workspace validation report.
        repair_diffs: Repair diffs the debug loop produced (audit trail).
        artifact_paths: Every artefact written, for ``artifact_written``
            events.
        experiment_workspace_path: Root of the materialized workspace.
        workflow_yaml_path: Path of ``ir/workflow.yaml``.
        source_root: Root of the generated ``src/`` tree.
        entrypoint_module: Dotted module of the workflow entrypoint.
        entrypoint_symbol: Name of the entrypoint callable.
    """

    model_config = _FROZEN

    plan_graph: PlanGraph
    codegen_ok: bool
    validation_report: ValidationReport
    repair_diffs: tuple[PlanDiff, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    experiment_workspace_path: Path
    workflow_yaml_path: Path
    source_root: Path
    entrypoint_module: str = _ENTRYPOINT_MODULE
    entrypoint_symbol: str = _ENTRYPOINT_SYMBOL


async def materialize_plan(
    *,
    harness: AgentHarness,
    handoff: object,
    layout: MaterializedLayout,
    config: object,
) -> MaterializationOutcome:
    """Run the codegen + debug + validation pipeline; return the outcome.

    Args:
        harness: The driving :class:`AgentHarness` (stages, router,
            execution env).
        handoff: The :class:`~molexp.agent.modes.plan.handoff.ApprovedPlanHandoff`.
        layout: The :class:`MaterializedLayout` over the plan folder.
        config: The :class:`~molexp.agent.modes.author._mode.AuthorModeConfig`.
    """
    from molexp.agent.modes.author._mode import AuthorModeConfig
    from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff

    assert isinstance(handoff, ApprovedPlanHandoff)
    assert isinstance(config, AuthorModeConfig)

    artifacts: list[str] = []
    router = harness.router

    # Stage 1 — LowerPlanGraph.
    async with harness.stage("LowerPlanGraph"):
        lowering = lower_plan_graph(handoff.plan_graph)
        write_workflow_ir(lowering.contract, layout=layout)
        artifacts.append(str(layout.workflow_yaml_path()))
    contract = lowering.contract
    plan_graph = lowering.plan_graph

    if not lowering.ok:
        return _failed_outcome(
            plan_graph=plan_graph,
            layout=layout,
            validation_report=lowering.validation_report,
            artifacts=artifacts,
        )

    # Stage 2 — CompileTaskIR.
    async with harness.stage("CompileTaskIR"):
        briefs = await compile_task_ir(
            router=router,
            contract=contract,
            capability_graph=handoff.capability_graph,
            layout=layout,
            tier=config.codegen_tier,
        )
        artifacts.extend(str(layout.task_ir_path(b.task_id)) for b in briefs)

    # Stage 3 — GenerateWorkflowSkeleton.
    try:
        async with harness.stage("GenerateWorkflowSkeleton"):
            workflow_py = generate_workflow_skeleton(contract, layout=layout)
            artifacts.append(workflow_py)
    except CodegenError:
        return _failed_outcome(
            plan_graph=plan_graph,
            layout=layout,
            validation_report=lowering.validation_report,
            artifacts=artifacts,
        )

    # Stage 4 — GenerateTaskTests / GenerateTaskImplementations.
    try:
        async with harness.stage("GenerateTaskTests"):
            test_paths = await generate_task_tests(
                router=router,
                briefs=briefs,
                contract=contract,
                capability_graph=handoff.capability_graph,
                layout=layout,
                tier=config.codegen_tier,
            )
            artifacts.extend(test_paths)
        async with harness.stage("GenerateTaskImplementations"):
            impl_paths = await generate_task_implementations(
                router=router,
                briefs=briefs,
                contract=contract,
                capability_graph=handoff.capability_graph,
                layout=layout,
                tier=config.codegen_tier,
            )
            artifacts.extend(impl_paths)
    except CodegenError as exc:
        diffs = _diffs_from_codegen_error(exc, plan_graph)
        return _failed_outcome(
            plan_graph=plan_graph,
            layout=layout,
            validation_report=lowering.validation_report,
            artifacts=artifacts,
            repair_diffs=diffs,
        )

    # Stage 5 — RunTaskDebugLoop.
    repair_diffs: list[PlanDiff] = []
    debug_ok = True
    async with harness.stage("RunTaskDebugLoop"):
        for brief in briefs:
            result = await run_task_debug_loop(
                task_id=brief.task_id,
                impl_path=layout.task_impl_path(brief.task_id),
                test_path=layout.task_test_path(brief.task_id),
                plan_graph=plan_graph,
                router=router,
                execution_env=harness.execution_env,
                src_root=layout.src_dir(),
                debug_attempts=config.debug_attempts,
                timeout=config.subprocess_timeout_seconds,
                tier=config.codegen_tier,
            )
            repair_diffs.extend(result.diffs)
            if not result.converged:
                debug_ok = False

    # Stage 6 — ValidateWorkspace.
    async with harness.stage("ValidateWorkspace"):
        validation = validate_workspace(contract)
        validation_report = lowering.validation_report
        layout.write(
            layout.validation_report_path(),
            validation_report.model_dump_json(indent=2),
        )
        artifacts.append(str(layout.validation_report_path()))

    # Stage 7 — WriteManifest.
    async with harness.stage("WriteManifest"):
        manifest_path = _write_manifest(
            layout=layout,
            plan_graph=plan_graph,
            task_ids=tuple(b.task_id for b in briefs),
            codegen_ok=debug_ok and validation.passed,
        )
        artifacts.append(str(manifest_path))

    return MaterializationOutcome(
        plan_graph=plan_graph,
        codegen_ok=debug_ok,
        validation_report=validation_report,
        repair_diffs=tuple(repair_diffs),
        artifact_paths=tuple(artifacts),
        experiment_workspace_path=layout.root(),
        workflow_yaml_path=layout.workflow_yaml_path(),
        source_root=layout.src_dir(),
    )


def _failed_outcome(
    *,
    plan_graph: PlanGraph,
    layout: MaterializedLayout,
    validation_report: ValidationReport,
    artifacts: list[str],
    repair_diffs: tuple[PlanDiff, ...] = (),
) -> MaterializationOutcome:
    """Build a failed :class:`MaterializationOutcome`."""
    return MaterializationOutcome(
        plan_graph=plan_graph,
        codegen_ok=False,
        validation_report=validation_report,
        repair_diffs=repair_diffs,
        artifact_paths=tuple(artifacts),
        experiment_workspace_path=layout.root(),
        workflow_yaml_path=layout.workflow_yaml_path(),
        source_root=layout.src_dir(),
    )


def _diffs_from_codegen_error(exc: CodegenError, plan_graph: PlanGraph) -> tuple[PlanDiff, ...]:
    """Translate a :class:`CodegenError` into repair diffs (one per miss)."""
    from molexp.agent.modes.author.repair import build_repair_diff

    if not exc.missing:
        return ()
    # All misses share the codegen failure; attribute to the first plan step.
    step_id = plan_graph.steps[0].id if plan_graph.steps else "workflow"
    return (
        build_repair_diff(
            plan_graph=plan_graph,
            step_id=step_id,
            traceback=str(exc),
            attempt=1,
        ),
    )


def _write_manifest(
    *,
    layout: MaterializedLayout,
    plan_graph: PlanGraph,
    task_ids: tuple[str, ...],
    codegen_ok: bool,
) -> Path:
    """Write ``manifest.yaml`` and return its path."""
    manifest = {
        "plan_id": plan_graph.plan_id,
        "compiled_contract_ref": plan_graph.compiled_contract_ref,
        "task_ids": list(task_ids),
        "entrypoint_module": _ENTRYPOINT_MODULE,
        "entrypoint_symbol": _ENTRYPOINT_SYMBOL,
        "status": "ready_for_run" if codegen_ok else "failed",
    }
    layout.write(layout.manifest_path(), yaml.safe_dump(manifest, sort_keys=False))
    return layout.manifest_path()
