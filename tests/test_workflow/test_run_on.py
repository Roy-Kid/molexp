"""``WorkflowRuntime.run_on(experiment, ...)`` convenience one-liner.

Covers ``oop-api-rectification`` ac-007: ``run_on`` wraps build-run-execute into
one call — it creates a fresh ``Run`` under the experiment, returns a
``WorkflowResult``, does NOT auto-bind the workflow to the experiment (binding is
the caller's choice), and on failure re-raises a ``RuntimeError`` carrying the
workflow name + final status while recording the run as failed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from molexp.workflow import (
    CompiledWorkflow,
    WorkflowCompiler,
    WorkflowRuntime,
    default_binding_registry,
)
from molexp.workspace import RunStatus, Workspace

if TYPE_CHECKING:
    from molexp.workflow.context import TaskContext


@pytest.fixture(autouse=True)
def _isolate_registry():
    default_binding_registry.clear()
    yield
    default_binding_registry.clear()


def _trivial_workflow() -> CompiledWorkflow:
    builder = WorkflowCompiler(name="trivial")

    @builder.task
    async def emit(ctx: TaskContext[None, None, None]) -> int:
        return 42

    return builder.compile()


def _failing_workflow() -> CompiledWorkflow:
    builder = WorkflowCompiler(name="failing")

    @builder.task
    async def boom(ctx: TaskContext[None, None, None]) -> None:
        raise RuntimeError("intentional failure")

    return builder.compile()


class TestRunOn:
    @pytest.mark.asyncio
    async def test_executes_and_creates_a_fresh_run(self, tmp_path):
        """run_on runs the workflow against a fresh Run and returns its result."""
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="demo").add_experiment(name="trivial-exp")

        runs_before = exp.list_runs()
        result = await WorkflowRuntime().run_on(_trivial_workflow(), exp, parameters={"lr": 1e-3})

        assert result.outputs.get("emit") == 42
        assert len(exp.list_runs()) == len(runs_before) + 1

    @pytest.mark.asyncio
    async def test_does_not_auto_bind(self, tmp_path):
        """run_on must NOT auto-bind the workflow — that is ``bind_to``'s job."""
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="demo").add_experiment(name="trivial-exp")

        assert default_binding_registry.for_experiment(exp) is None
        await WorkflowRuntime().run_on(_trivial_workflow(), exp)
        assert default_binding_registry.for_experiment(exp) is None

    @pytest.mark.asyncio
    async def test_reraises_and_records_failed_run(self, tmp_path):
        """A task failure re-raises a RuntimeError naming the workflow + status,
        and the created run is left FAILED."""
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="demo").add_experiment(name="failing-exp")

        with pytest.raises(RuntimeError, match=r"failing.*status 'failed'"):
            await WorkflowRuntime().run_on(_failing_workflow(), exp)

        runs = exp.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED
