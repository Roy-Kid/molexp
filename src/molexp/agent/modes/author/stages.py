"""AuthorMode's eight Stage subclasses.

Each Stage wraps one ``async with harness.stage("X"): ...`` block from
the original ``materialize_plan`` function. State flows through
mutable :class:`AuthorMode` attributes the stages read / write — the
``_StageOutcome`` pattern remains pragmatic for this mode given its
eight tightly-coupled stages.

Order:

1. :class:`LowerPlanGraph` — typed ``PlanGraph`` → ``WorkflowContract``
2. :class:`CompileTaskIR` — per-task IR briefs
3. :class:`GenerateWorkflowSkeleton` — experiment package skeleton
4. :class:`GenerateTaskTests` — per-task tests
5. :class:`GenerateTaskImplementations` — per-task implementations
6. :class:`RunTaskDebugLoop` — isolated-subprocess run+repair loop
7. :class:`ValidateWorkspace` — re-validate the materialized IR
8. :class:`WriteManifest` — write ``manifest.yaml``

Any failure short-circuits later stages via ``author_mode._failed``;
:meth:`AuthorMode.run` reads the accumulated state post-pipeline to
build the terminal completion event.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

import yaml

from molexp.agent.events import AgentEvent
from molexp.agent.modes.author.codegen import (
    CodegenError,
    TaskIRBrief,
    compile_task_ir,
    generate_task_implementations,
    generate_task_tests,
    generate_workflow_skeleton,
    validate_workspace,
    write_workflow_ir,
)
from molexp.agent.modes.author.debug_loop import DebugLoopResult, run_task_debug_loop
from molexp.agent.modes.author.lowering import lower_plan_graph
from molexp.agent.stage import Stage

if TYPE_CHECKING:
    from molexp.agent.modes.author._mode import AuthorMode
    from molexp.agent.runtime import AgentHarness

__all__ = [
    "CompileTaskIR",
    "GenerateTaskImplementations",
    "GenerateTaskTests",
    "GenerateWorkflowSkeleton",
    "LowerPlanGraph",
    "RunTaskDebugLoop",
    "ValidateWorkspace",
    "WriteManifest",
]

_ENTRYPOINT_MODULE = "experiment.workflow"
_ENTRYPOINT_SYMBOL = "create_workflow"


def _skip_if_failed(mode: AuthorMode) -> bool:
    """Return ``True`` when an earlier stage marked the run as failed."""
    return bool(mode._failed)


class LowerPlanGraph(Stage[object, object]):
    """Stage 1 — lower the typed PlanGraph into a WorkflowContract."""

    name: ClassVar[str] = "LowerPlanGraph"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode) or mode._injected_handoff is None:
            yield None
            return
        lowering = lower_plan_graph(mode._injected_handoff.plan_graph)
        layout = mode._layout()
        write_workflow_ir(lowering.contract, layout=layout)
        mode._artifacts.append(str(layout.workflow_yaml_path()))
        mode._lowering = lowering
        mode._contract = lowering.contract
        mode._plan_graph = lowering.plan_graph
        if not lowering.ok:
            mode._failed = True
            mode._validation_report = lowering.validation_report
        yield None


class CompileTaskIR(Stage[object, object]):
    """Stage 2 — draft one IR brief per task."""

    name: ClassVar[str] = "CompileTaskIR"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._contract is not None
        assert mode._injected_handoff is not None
        layout = mode._layout()
        briefs = await compile_task_ir(
            router=harness.router,
            contract=mode._contract,
            plan_graph=mode._injected_handoff.plan_graph,
            layout=layout,
            tier=mode.config.codegen_tier,
        )
        mode._briefs = briefs
        mode._artifacts.extend(str(layout.task_ir_path(b.task_id)) for b in briefs)
        yield None


class GenerateWorkflowSkeleton(Stage[object, object]):
    """Stage 3 — emit the experiment package skeleton."""

    name: ClassVar[str] = "GenerateWorkflowSkeleton"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._contract is not None
        layout = mode._layout()
        try:
            workflow_py = generate_workflow_skeleton(mode._contract, layout=layout)
        except CodegenError:
            mode._failed = True
            yield None
            return
        mode._artifacts.append(workflow_py)
        yield None


class GenerateTaskTests(Stage[object, object]):
    """Stage 4 — per-task test files."""

    name: ClassVar[str] = "GenerateTaskTests"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._contract is not None
        assert mode._injected_handoff is not None
        layout = mode._layout()
        try:
            paths = await generate_task_tests(
                router=harness.router,
                briefs=mode._briefs,
                contract=mode._contract,
                plan_graph=mode._injected_handoff.plan_graph,
                layout=layout,
                tier=mode.config.codegen_tier,
            )
        except CodegenError as exc:
            mode._failed = True
            mode._codegen_error = exc
            yield None
            return
        mode._artifacts.extend(paths)
        yield None


class GenerateTaskImplementations(Stage[object, object]):
    """Stage 5 — per-task implementations."""

    name: ClassVar[str] = "GenerateTaskImplementations"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._contract is not None
        assert mode._injected_handoff is not None
        layout = mode._layout()
        try:
            paths = await generate_task_implementations(
                router=harness.router,
                briefs=mode._briefs,
                contract=mode._contract,
                plan_graph=mode._injected_handoff.plan_graph,
                layout=layout,
                tier=mode.config.codegen_tier,
            )
        except CodegenError as exc:
            mode._failed = True
            mode._codegen_error = exc
            yield None
            return
        mode._artifacts.extend(paths)
        yield None


class RunTaskDebugLoop(Stage[object, object]):
    """Stage 6 — isolated-subprocess run+repair loop for each task."""

    name: ClassVar[str] = "RunTaskDebugLoop"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._plan_graph is not None
        layout = mode._layout()
        repair = mode._build_repair_callable()
        # Per-task debug loops are independent (each task's test + repair
        # touches only its own impl_path / test_path), so run them
        # concurrently — pre-rewrite this was a serial Python ``for``
        # that took N_tasks x debug_attempts x (subprocess + LLM-repair)
        # wall-clock, which on the smoke prompt was the 59-min tail
        # dwarfing every other stage.
        plan_graph = mode._plan_graph

        async def _run_one(brief: TaskIRBrief) -> DebugLoopResult:
            return await run_task_debug_loop(
                task_id=brief.task_id,
                impl_path=layout.task_impl_path(brief.task_id),
                test_path=layout.task_test_path(brief.task_id),
                plan_graph=plan_graph,
                router=harness.router,
                execution_env=harness.execution_env,
                src_root=layout.src_dir(),
                debug_attempts=mode.config.debug_attempts,
                timeout=mode.config.subprocess_timeout_seconds,
                tier=mode.config.codegen_tier,
                repair=repair,
            )

        results = await asyncio.gather(*(_run_one(b) for b in mode._briefs))
        debug_ok = True
        for result in results:
            mode._repair_diffs.extend(result.diffs)
            if not result.converged:
                debug_ok = False
        mode._debug_ok = debug_ok
        yield None


class ValidateWorkspace(Stage[object, object]):
    """Stage 7 — re-validate the materialized IR."""

    name: ClassVar[str] = "ValidateWorkspace"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._contract is not None
        assert mode._lowering is not None
        layout = mode._layout()
        validation = validate_workspace(mode._contract)
        # Persist the lowering's validation report (matches original behavior).
        validation_report = mode._lowering.validation_report
        layout.write(
            layout.validation_report_path(),
            validation_report.model_dump_json(indent=2),
        )
        mode._artifacts.append(str(layout.validation_report_path()))
        mode._validation_report = validation_report
        mode._validation_passed = validation.passed
        yield None


class WriteManifest(Stage[object, object]):
    """Stage 8 — write ``manifest.yaml``."""

    name: ClassVar[str] = "WriteManifest"

    def __init__(self, *, author_mode: AuthorMode) -> None:
        self._mode = author_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if _skip_if_failed(mode):
            yield None
            return
        assert mode._plan_graph is not None
        layout = mode._layout()
        codegen_ok = mode._debug_ok and mode._validation_passed
        manifest = {
            "plan_id": mode._plan_graph.plan_id,
            "compiled_contract_ref": mode._plan_graph.compiled_contract_ref,
            "task_ids": [b.task_id for b in mode._briefs],
            "entrypoint_module": _ENTRYPOINT_MODULE,
            "entrypoint_symbol": _ENTRYPOINT_SYMBOL,
            "status": "ready_for_run" if codegen_ok else "failed",
        }
        layout.write(layout.manifest_path(), yaml.safe_dump(manifest, sort_keys=False))
        mode._artifacts.append(str(layout.manifest_path()))
        mode._manifest_written = True
        yield None
