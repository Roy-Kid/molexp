"""Tests for ``GenerateTaskTests`` task.

Covers ac-003: one test file per task + topology-pin test; the
``is_stub=True`` path emits ``pytest.skip("stub")``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.agent.modes.plan.policy import PlanModelPolicy
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    SkeletonResult,
    TaskIRBrief,
    TaskIRResult,
    TaskTestsResult,
)
from molexp.agent.modes.plan.tasks import GenerateTaskTests
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace

from .conftest import FakeProvider


@pytest.fixture
def gen_tests_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="gen_tests")


@pytest.mark.asyncio
async def test_generate_task_tests_emits_three_modules_plus_structure(
    gen_tests_handle: PlanWorkspaceHandle,
) -> None:
    fake = FakeProvider()
    deps = PlanDeps(
        router=fake,
        policy=PlanModelPolicy(),
        workspace_handle=gen_tests_handle,
    )
    briefs = (
        TaskIRBrief(task_id="prepare", responsibility="weigh"),
        TaskIRBrief(task_id="couple", responsibility="reflux"),
        TaskIRBrief(task_id="isolate", responsibility="filter"),
    )
    ctx = TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=tuple(Path(f"ir/tasks/{b.task_id}.yaml") for b in briefs),
                briefs=briefs,
            ),
            "GenerateWorkflowSkeleton": SkeletonResult(
                workflow_py_path=gen_tests_handle.experiment_pkg_dir() / "workflow.py",
                package_path=gen_tests_handle.experiment_pkg_dir(),
            ),
        },
        config={},
    )
    result = await GenerateTaskTests().execute(ctx)
    assert isinstance(result, TaskTestsResult)
    assert len(result.test_paths) == 3

    tests_dir = gen_tests_handle.tests_dir()
    for task_id in ("prepare", "couple", "isolate"):
        path = tests_dir / f"test_{task_id}.py"
        assert path.exists()
    # Topology-pin test landed.
    structure = tests_dir / "test_workflow_structure.py"
    assert structure.exists()
    structure_text = structure.read_text()
    assert "default_compiler.dict_to_contract" in structure_text or "yaml_to_ir" in structure_text


@pytest.mark.asyncio
async def test_generate_task_tests_stub_emits_pytest_skip(
    gen_tests_handle: PlanWorkspaceHandle,
) -> None:
    fake = FakeProvider()
    deps = PlanDeps(
        router=fake,
        policy=PlanModelPolicy(),
        workspace_handle=gen_tests_handle,
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
                workflow_py_path=gen_tests_handle.experiment_pkg_dir() / "workflow.py",
                package_path=gen_tests_handle.experiment_pkg_dir(),
            ),
        },
        config={},
    )
    await GenerateTaskTests().execute(ctx)
    test_text = (gen_tests_handle.tests_dir() / "test_stub_task.py").read_text()
    assert 'pytest.skip("stub")' in test_text
