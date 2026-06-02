"""``Workflow.run_on(experiment, ...)`` happy-path one-liner.

Covers acceptance criterion ac-007 of the ``oop-api-rectification``
spec: ``run_on`` wraps the build-run-execute three-step into one
call, runs the workflow against a fresh ``Run``, returns a
``WorkflowResult``, and **does not** auto-bind the workflow to the
experiment (binding is the caller's choice).
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


@pytest.mark.asyncio
async def test_run_on_returns_workflow_result(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="trivial-exp")
    wf = _trivial_workflow()

    result = await WorkflowRuntime().run_on(wf, exp)

    assert result is not None
    assert result.outputs.get("emit") == 42


@pytest.mark.asyncio
async def test_run_on_does_not_auto_bind(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="trivial-exp")
    wf = _trivial_workflow()

    assert default_binding_registry.for_experiment(exp) is None
    await WorkflowRuntime().run_on(wf, exp)
    # run_on must NOT auto-bind — that's bind_to's job.
    assert default_binding_registry.for_experiment(exp) is None


@pytest.mark.asyncio
async def test_run_on_creates_a_run_under_the_experiment(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="trivial-exp")
    wf = _trivial_workflow()

    runs_before = exp.list_runs()
    await WorkflowRuntime().run_on(wf, exp, parameters={"lr": 1e-3})
    runs_after = exp.list_runs()

    assert len(runs_after) == len(runs_before) + 1


@pytest.mark.asyncio
async def test_run_on_failure_propagates_and_records_failed_status(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="failing-exp")
    wf = _failing_workflow()

    # The underlying workflow runtime catches task exceptions and stores
    # the failure on the WorkflowResult.status; ``run_on`` re-raises a
    # RuntimeError carrying the workflow name + final status. Original
    # task traceback is preserved in the runtime logs but not on the
    # rebuilt exception.
    with pytest.raises(RuntimeError, match=r"failing.*status 'failed'"):
        await WorkflowRuntime().run_on(wf, exp)

    runs = exp.list_runs()
    assert len(runs) == 1
    assert runs[0].status == RunStatus.FAILED
