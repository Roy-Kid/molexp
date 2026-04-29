"""Unit tests for the native task-management tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.workspace_tools import (
    ask_user,
    get_run_status,
    retry_run,
    submit_run,
    wait_for_run,
)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_submit_run_creates_new_run(fake_ctx, experiment):
    out = await submit_run(
        fake_ctx,
        project_id=experiment.project.id,
        experiment_id=experiment.id,
        parameters={"temperature": 400, "seed": 2},
    )
    assert "run_id" in out
    assert out["parameters"] == {"temperature": 400, "seed": 2}
    # Run is on disk now.
    assert experiment.get_run(out["run_id"]) is not None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_submit_run_unknown_project_returns_error(fake_ctx):
    out = await submit_run(
        fake_ctx,
        project_id="missing",
        experiment_id="missing",
        parameters={},
    )
    assert "error" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_run_status_returns_metadata(fake_ctx, experiment, existing_run):
    out = await get_run_status(
        fake_ctx,
        project_id=experiment.project.id,
        experiment_id=experiment.id,
        run_id=existing_run.id,
    )
    assert out["run_id"] == existing_run.id
    assert "status" in out
    assert "started_at" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_run_status_missing_run(fake_ctx, experiment):
    out = await get_run_status(
        fake_ctx,
        project_id=experiment.project.id,
        experiment_id=experiment.id,
        run_id="does-not-exist",
    )
    assert "error" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wait_for_run_times_out_when_not_terminal(fake_ctx, experiment, existing_run):
    out = await wait_for_run(
        fake_ctx,
        project_id=experiment.project.id,
        experiment_id=experiment.id,
        run_id=existing_run.id,
        timeout_seconds=0.2,
        poll_interval=0.1,
    )
    assert out.get("timed_out") is True
    assert out.get("run_id") == existing_run.id


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retry_run_clones_parameters(fake_ctx, experiment, existing_run):
    out = await retry_run(
        fake_ctx,
        project_id=experiment.project.id,
        experiment_id=experiment.id,
        run_id=existing_run.id,
    )
    assert out["source_run_id"] == existing_run.id
    assert out["new_run_id"] != existing_run.id
    new_run = experiment.get_run(out["new_run_id"])
    assert new_run is not None
    # Parameters carried over from the original.
    assert dict(new_run.parameters) == dict(existing_run.parameters)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ask_user_routes_through_session(workspace):
    class _StubSession:
        def __init__(self) -> None:
            self.last_prompt: str | None = None

        async def await_user_message(self, prompt: str) -> str:
            self.last_prompt = prompt
            return "okay scope=project"

    session = _StubSession()
    deps = SimpleNamespace(workspace=workspace, session=session, current_run=None)
    ctx = SimpleNamespace(deps=deps)

    out = await ask_user(ctx, prompt="What scope?")
    assert out == {"content": "okay scope=project"}
    assert session.last_prompt == "What scope?"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ask_user_without_session_returns_error(workspace):
    deps = SimpleNamespace(workspace=workspace, session=None, current_run=None)
    ctx = SimpleNamespace(deps=deps)
    out = await ask_user(ctx, prompt="anything")
    assert "error" in out


def test_event_loop_smoke():
    """Pytest-asyncio plugin sanity check (catches missing pytest-asyncio early)."""
    assert asyncio.iscoroutinefunction(submit_run)
