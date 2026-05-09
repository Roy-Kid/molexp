"""Tests for ``ValidateWorkspace`` task.

Covers ac-005: severity matrix — missing files / parse failures /
import failures are errors that fail the pass; warnings (e.g. empty
contract) do not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.agent.modes.plan.policy import PlanModelPolicy
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    TaskImplementationsResult,
    TaskIRBrief,
    TaskIRResult,
    TaskTestsResult,
    ValidationResult,
)
from molexp.agent.modes.plan.tasks import ValidateWorkspace
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace


def _make_ctx(
    handle: PlanWorkspaceHandle,
    *,
    briefs: tuple[TaskIRBrief, ...],
) -> TaskContext:
    deps = PlanDeps(
        provider=_NoopProvider(),  # type: ignore[arg-type]
        policy=PlanModelPolicy(),
        workspace_handle=handle,
    )
    test_paths = tuple(handle.tests_dir() / f"test_{b.task_id}.py" for b in briefs)
    impl_paths = tuple(handle.tasks_pkg_dir() / f"{b.task_id}.py" for b in briefs)
    return TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=tuple(Path(f"ir/tasks/{b.task_id}.yaml") for b in briefs),
                briefs=briefs,
            ),
            "GenerateTaskTests": TaskTestsResult(test_paths=test_paths),
            "GenerateTaskImplementations": TaskImplementationsResult(impl_paths=impl_paths),
        },
        config={},
    )


class _NoopProvider:
    """Validate doesn't call the provider, but PlanDeps requires one."""

    async def invoke(self, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("ValidateWorkspace must not call provider")

    async def invoke_with_template(self, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("ValidateWorkspace must not call provider")


@pytest.fixture
def valid_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    """Workspace seeded with all the artifacts ValidateWorkspace expects."""
    handle = PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="valid_ws")
    # Required paths (the validator checks each for existence).
    (handle.report_dir() / "original.md").write_text("report")
    (handle.plan_dir() / "implementation_plan.md").write_text("plan")
    (handle.experiment_pkg_dir() / "__init__.py").write_text("")
    (handle.experiment_pkg_dir() / "workflow.py").write_text(
        "from molexp.workflow import WorkflowBuilder\n"
        "WORKFLOW = WorkflowBuilder(name='wf').build()\n"
    )
    (handle.tasks_pkg_dir() / "__init__.py").write_text("")
    (handle.ir_dir() / "workflow.yaml").write_text("workflow_id: wf\ntask_io: []\n")
    return handle


@pytest.mark.asyncio
async def test_validate_workspace_passes_when_all_artifacts_present(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    briefs = (
        TaskIRBrief(task_id="prepare", responsibility="weigh"),
        TaskIRBrief(task_id="couple", responsibility="reflux"),
    )
    # Lay down the per-task impls + tests the briefs claim exist.
    for b in briefs:
        (valid_handle.tasks_pkg_dir() / f"{b.task_id}.py").write_text("")
        (valid_handle.tests_dir() / f"test_{b.task_id}.py").write_text("")

    ctx = _make_ctx(valid_handle, briefs=briefs)
    result = await ValidateWorkspace().execute(ctx)
    assert isinstance(result, ValidationResult)
    assert result.passed is True
    assert valid_handle.validation_report_path().exists()


@pytest.mark.asyncio
async def test_validate_workspace_fails_when_impl_missing(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    briefs = (TaskIRBrief(task_id="missing_impl", responsibility="x"),)
    # Test file present but impl module missing — that should be an
    # error severity check failure.
    (valid_handle.tests_dir() / "test_missing_impl.py").write_text("")

    ctx = _make_ctx(valid_handle, briefs=briefs)
    result = await ValidateWorkspace().execute(ctx)
    assert result.passed is False
    report_text = valid_handle.validation_report_path().read_text()
    assert "impl_present[missing_impl]" in report_text


@pytest.mark.asyncio
async def test_validate_workspace_fails_when_test_missing(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    briefs = (TaskIRBrief(task_id="missing_test", responsibility="x"),)
    (valid_handle.tasks_pkg_dir() / "missing_test.py").write_text("")

    ctx = _make_ctx(valid_handle, briefs=briefs)
    result = await ValidateWorkspace().execute(ctx)
    assert result.passed is False


@pytest.mark.asyncio
async def test_validate_workspace_fails_on_invalid_workflow_py(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    """A workflow.py that does not parse should surface a compile error."""
    (valid_handle.experiment_pkg_dir() / "workflow.py").write_text("def def def\n")

    ctx = _make_ctx(valid_handle, briefs=())
    result = await ValidateWorkspace().execute(ctx)
    assert result.passed is False


@pytest.mark.asyncio
async def test_validate_workspace_warns_on_empty_contract(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    ctx = _make_ctx(valid_handle, briefs=())
    result = await ValidateWorkspace().execute(ctx)
    assert result.passed is True
    text = valid_handle.validation_report_path().read_text()
    assert "no tasks declared" in text


@pytest.mark.asyncio
async def test_validate_workspace_writes_validation_report(
    valid_handle: PlanWorkspaceHandle,
) -> None:
    briefs = (TaskIRBrief(task_id="t", responsibility="r"),)
    (valid_handle.tasks_pkg_dir() / "t.py").write_text("")
    (valid_handle.tests_dir() / "test_t.py").write_text("")
    ctx = _make_ctx(valid_handle, briefs=briefs)
    await ValidateWorkspace().execute(ctx)
    md = valid_handle.validation_report_path().read_text()
    # Markdown header + table.
    assert md.startswith("# Validation report")
    assert "passed" in md or "failed" in md
