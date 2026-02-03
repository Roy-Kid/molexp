"""Tests for Task submission protocol."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock

import pytest
from pydantic import BaseModel

from molexp.workflow.task import Task
from molexp.workflow.engine import WorkflowEngine
from molexp.workflow.workflow import Workflow
from molexp.workspace import Workspace


# ============================================================================
# Test Helpers
# ============================================================================


def _make_hierarchy(tmp_path):
    workspace = Workspace(root=tmp_path / "workspace", name="Test")
    workspace.materialize()
    project = workspace.create_project(name="Project")
    experiment = project.create_experiment(name="Experiment")
    return workspace, project, experiment


def _make_run(tmp_path, parameters=None):
    _, _, experiment = _make_hierarchy(tmp_path)
    return experiment.create_run(parameters=parameters or {})


# ============================================================================
# Test Tasks
# ============================================================================


class SimpleConfig(BaseModel):
    value: int = 42


class LocalTask(Task[SimpleConfig, dict]):
    """Task without submittable - executes locally."""

    config_type = SimpleConfig
    inputs = {}
    outputs = {"result": int}
    submittable = None  # Explicit None for local execution

    def execute(self, ctx=None, **inputs):
        return {"result": self.config.value * 2}


class SubmittableTask(Task[SimpleConfig, dict]):
    """Task with submittable - submits via molq."""

    config_type = SimpleConfig
    inputs = {}
    outputs = {"result": int, "job_id": str}
    submittable = "molq"

    def execute(self, ctx=None, **inputs):
        # Fallback for local execution
        return {"result": self.config.value * 2, "job_id": "local"}

    def submit(self, ctx=None, **inputs) -> Generator[dict, int, dict]:
        """Submit job via generator protocol."""
        # Create JobSpec (using dict for simplicity in tests)
        job_spec = {
            "execution": {"command": f"echo {self.config.value}"},
            "resources": {"cpu_count": 1},
        }

        # Yield JobSpec, receive job_id
        job_id = yield job_spec

        # Return results with job_id
        return {
            "result": self.config.value * 2,
            "job_id": job_id,
        }


class MissingSubmitTask(Task[SimpleConfig, dict]):
    """Task with submittable but no submit() implementation."""

    config_type = SimpleConfig
    inputs = {}
    outputs = {"result": int}
    submittable = "molq"

    def execute(self, ctx=None, **inputs):
        return {"result": 0}

    # Does not implement submit()


# ============================================================================
# Tests
# ============================================================================


def test_task_submittable_class_attribute():
    """Test that submittable can be set as class attribute."""
    assert LocalTask.submittable is None
    assert SubmittableTask.submittable == "molq"

    # Instance should inherit class attribute
    local = LocalTask()
    assert local.submittable is None

    submittable = SubmittableTask()
    assert submittable.submittable == "molq"


def test_task_submittable_instance_override():
    """Test that instance can override class-level submittable."""
    task = LocalTask()
    assert task.submittable is None

    # Override at instance level
    task.submittable = "custom"
    assert task.submittable == "custom"
    assert LocalTask.submittable is None  # Class unchanged


def test_submit_not_implemented_error():
    """Test that submit() raises NotImplementedError with clear message."""
    task = MissingSubmitTask()

    with pytest.raises(NotImplementedError, match="submittable='molq'"):
        # Trigger generator
        gen = task.submit()
        next(gen)


def test_local_task_execution(tmp_path):
    """Test that task with submittable=None executes locally."""
    task = LocalTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])

    engine = WorkflowEngine(workflow)
    run = _make_run(tmp_path)

    with run.start() as ctx:
        results = engine.execute(ctx)

    assert task.task_id in results
    assert results[task.task_id]["result"] == 20


def test_submittable_task_without_submitor_fails(tmp_path):
    """Test that submittable task without registered submitor raises error."""
    task = SubmittableTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])

    engine = WorkflowEngine(workflow)
    run = _make_run(tmp_path)

    # Error is caught by workflow engine and stored, not raised
    with run.start() as ctx:
        results = engine.execute(ctx)

    # Task should have failed
    assert task.task_id not in results
    assert task.task_id in ctx.context.errors
    error = ctx.context.errors[task.task_id]
    assert "Submitor backend 'molq' not configured" in error["message"]


def test_submitor_registration():
    """Test that submitors can be registered."""
    workflow = Workflow.from_tasks(tasks=[], links=[])
    engine = WorkflowEngine(workflow)

    # Create mock submitor
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=123)
    mock_submitor.query = Mock(return_value={})
    mock_submitor.cancel = Mock()

    engine.register_submitor("molq", mock_submitor)
    assert "molq" in engine._submitors
    assert engine._submitors["molq"] == mock_submitor


def test_generator_protocol_handling(tmp_path):
    """Test that engine handles generator protocol correctly."""
    task = SubmittableTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])

    engine = WorkflowEngine(workflow)

    # Mock submitor that returns a job_id
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=123)
    mock_submitor.query = Mock(return_value={})

    engine._submitors["molq"] = mock_submitor

    run = _make_run(tmp_path)

    with run.start() as ctx:
        results = engine.execute(ctx)

    # Verify submission happened
    assert mock_submitor.submit.called
    job_spec = mock_submitor.submit.call_args[0][0]
    assert "execution" in job_spec
    assert "command" in job_spec["execution"]

    # Verify results
    assert task.task_id in results
    assert results[task.task_id]["result"] == 20
    assert results[task.task_id]["job_id"] == 123


def test_submission_tracking_in_context(tmp_path):
    """Test that submissions are tracked in RunContext."""
    task = SubmittableTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])

    engine = WorkflowEngine(workflow)

    # Mock submitor
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=456)
    mock_submitor.query = Mock(return_value={})

    engine._submitors["molq"] = mock_submitor

    run = _make_run(tmp_path)

    with run.start() as ctx:
        engine.execute(ctx)

        # Check submissions were tracked in context.execution
        assert "submissions" in ctx.context.execution
        assert task.task_id in ctx.context.execution["submissions"]
        submissions = ctx.context.execution["submissions"][task.task_id]
        assert len(submissions) == 1
        assert submissions[0]["job_id"] == 456
        assert submissions[0]["backend"] == "molq"


def test_mixed_local_and_submittable_tasks(tmp_path):
    """Test workflow with both local and submittable tasks."""
    local_task = LocalTask(value=5)
    submit_task = SubmittableTask(value=10)

    workflow = Workflow.from_tasks(tasks=[local_task, submit_task], links=[])

    engine = WorkflowEngine(workflow)

    # Mock submitor
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=789)
    mock_submitor.query = Mock(return_value={})
    engine._submitors["molq"] = mock_submitor

    run = _make_run(tmp_path)

    with run.start() as ctx:
        results = engine.execute(ctx)

    # Both tasks executed
    assert local_task.task_id in results
    assert submit_task.task_id in results

    # Local task executed directly
    assert results[local_task.task_id]["result"] == 10

    # Submittable task went through submitor
    assert mock_submitor.submit.called
    assert results[submit_task.task_id]["result"] == 20
    assert results[submit_task.task_id]["job_id"] == 789


def test_job_monitoring_with_blocking(tmp_path):
    """Test that blocking jobs are monitored until completion."""
    from molq.resources import JobSpec, ExecutionSpec, ResourceSpec
    from molq.jobstatus import JobStatus

    # Create task that submits with blocking enabled
    class BlockingTask(Task[SimpleConfig, dict]):
        config_type = SimpleConfig
        inputs = {}
        outputs = {"result": int, "job_id": int}
        submittable = "molq"

        def execute(self, ctx=None, **inputs):
            return {"result": 0, "job_id": 0}

        def submit(self, ctx=None, **inputs) -> Generator[dict, int, dict]:
            job_spec = JobSpec(
                execution=ExecutionSpec(
                    cmd="echo test",
                    block=True  # Enable blocking
                ),
                resources=ResourceSpec(cpu_count=1),
            )
            job_id = yield job_spec
            return {"result": self.config.value * 2, "job_id": job_id}

    task = BlockingTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])
    engine = WorkflowEngine(workflow)

    # Mock submitor with status progression
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=999)

    # Simulate job lifecycle: PENDING -> RUNNING -> COMPLETED
    status_sequence = [
        {999: JobStatus(job_id=999, status=JobStatus.Status.PENDING, name="test_job")},
        {999: JobStatus(job_id=999, status=JobStatus.Status.RUNNING, name="test_job")},
        {999: JobStatus(job_id=999, status=JobStatus.Status.COMPLETED, name="test_job")},
    ]
    mock_submitor.query = Mock(side_effect=status_sequence)

    engine._submitors["molq"] = mock_submitor

    run = _make_run(tmp_path)

    with run.start() as ctx:
        results = engine.execute(ctx)

    # Verify job was submitted
    assert mock_submitor.submit.called

    # Verify query was called multiple times (monitoring)
    assert mock_submitor.query.call_count == 3

    # Verify task completed successfully
    assert task.task_id in results
    assert results[task.task_id]["job_id"] == 999


def test_job_monitoring_failure(tmp_path):
    """Test that failed jobs raise errors."""
    from molq.resources import JobSpec, ExecutionSpec, ResourceSpec
    from molq.jobstatus import JobStatus

    # Create task that submits with blocking enabled
    class FailingTask(Task[SimpleConfig, dict]):
        config_type = SimpleConfig
        inputs = {}
        outputs = {"result": int}
        submittable = "molq"

        def execute(self, ctx=None, **inputs):
            return {"result": 0}

        def submit(self, ctx=None, **inputs) -> Generator[dict, int, dict]:
            job_spec = JobSpec(
                execution=ExecutionSpec(cmd="exit 1", block=True),
                resources=ResourceSpec(cpu_count=1),
            )
            job_id = yield job_spec
            return {"result": self.config.value * 2}

    task = FailingTask(value=10)
    workflow = Workflow.from_tasks(tasks=[task], links=[])
    engine = WorkflowEngine(workflow)

    # Mock submitor with failure
    mock_submitor = Mock()
    mock_submitor.submit = Mock(return_value=888)
    mock_submitor.query = Mock(return_value={
        888: JobStatus(job_id=888, status=JobStatus.Status.FAILED, name="failing_job")
    })

    engine._submitors["molq"] = mock_submitor

    run = _make_run(tmp_path)

    # Task should fail
    with run.start() as ctx:
        results = engine.execute(ctx)

    # Task should have failed
    assert task.task_id not in results
    assert task.task_id in ctx.context.errors
    error = ctx.context.errors[task.task_id]
    assert "Job 888 failed" in error["message"]
