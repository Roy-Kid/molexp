"""Tests for ``GenerateTaskImplementations`` task.

Covers ac-004: full impl source for non-stub tasks; ``raise
NotImplementedError`` body for ``is_stub=True`` tasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.agent.modes.plan.policy import PlanModelPolicy
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    SkeletonResult,
    TaskImplementationModule,
    TaskImplementationsResult,
    TaskIRBrief,
    TaskIRResult,
)
from molexp.agent.modes.plan.tasks import GenerateTaskImplementations
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace

from .conftest import FakeProvider


@pytest.fixture
def impl_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="impl_gen")


@pytest.mark.asyncio
async def test_generate_impl_writes_llm_source_for_non_stub_tasks(
    impl_handle: PlanWorkspaceHandle,
) -> None:
    fake = FakeProvider()
    deps = PlanDeps(
        router=fake,
        policy=PlanModelPolicy(),
        workspace_handle=impl_handle,
    )
    briefs = (TaskIRBrief(task_id="prepare", responsibility="weigh"),)
    ctx = TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=(Path("ir/tasks/prepare.yaml"),),
                briefs=briefs,
            ),
            "GenerateWorkflowSkeleton": SkeletonResult(
                workflow_py_path=impl_handle.experiment_pkg_dir() / "workflow.py",
                package_path=impl_handle.experiment_pkg_dir(),
            ),
        },
        config={},
    )
    result = await GenerateTaskImplementations().execute(ctx)
    assert isinstance(result, TaskImplementationsResult)
    assert len(result.impl_paths) == 1

    impl_text = (impl_handle.tasks_pkg_dir() / "prepare.py").read_text()
    # The fake provider's canned source for `prepare` is a non-stub.
    assert "Generated implementation for prepare" in impl_text
    assert "raise NotImplementedError" not in impl_text


@pytest.mark.asyncio
async def test_generate_impl_writes_stub_body_for_is_stub_brief(
    impl_handle: PlanWorkspaceHandle,
) -> None:
    fake = FakeProvider()
    deps = PlanDeps(
        router=fake,
        policy=PlanModelPolicy(),
        workspace_handle=impl_handle,
    )
    briefs = (TaskIRBrief(task_id="stub_task", responsibility="todo", is_stub=True),)
    ctx = TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=(Path("ir/tasks/stub_task.yaml"),),
                briefs=briefs,
            ),
            "GenerateWorkflowSkeleton": SkeletonResult(
                workflow_py_path=impl_handle.experiment_pkg_dir() / "workflow.py",
                package_path=impl_handle.experiment_pkg_dir(),
            ),
        },
        config={},
    )
    await GenerateTaskImplementations().execute(ctx)
    impl_text = (impl_handle.tasks_pkg_dir() / "stub_task.py").read_text()
    assert "raise NotImplementedError" in impl_text


@pytest.mark.asyncio
async def test_generate_impl_writes_stub_body_when_module_marks_self_stub(
    impl_handle: PlanWorkspaceHandle,
) -> None:
    """The provider returns ``is_stub=True`` even though the brief did not
    mark the task as stub — the impl task respects the module-level flag
    and writes a NotImplementedError body."""
    presets_with_stub = {
        TaskImplementationModule: {
            "self_stub": TaskImplementationModule(
                task_id="self_stub",
                source="(this should be discarded)",
                is_stub=True,
            ),
        },
    }
    # Combine with the default presets so other schemas resolve too.
    from .conftest import canned_presets

    combined = canned_presets() | presets_with_stub  # type: ignore[operator]
    fake = FakeProvider(presets=combined)
    deps = PlanDeps(
        router=fake,
        policy=PlanModelPolicy(),
        workspace_handle=impl_handle,
    )
    briefs = (TaskIRBrief(task_id="self_stub", responsibility="meh"),)
    ctx = TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=(Path("ir/tasks/self_stub.yaml"),),
                briefs=briefs,
            ),
            "GenerateWorkflowSkeleton": SkeletonResult(
                workflow_py_path=impl_handle.experiment_pkg_dir() / "workflow.py",
                package_path=impl_handle.experiment_pkg_dir(),
            ),
        },
        config={},
    )
    await GenerateTaskImplementations().execute(ctx)
    impl_text = (impl_handle.tasks_pkg_dir() / "self_stub.py").read_text()
    assert "raise NotImplementedError" in impl_text
