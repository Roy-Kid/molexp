"""``drive_plan_mode`` — plan bundle inside the run lifecycle (honest status)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Workspace


class _FakeResult:
    pass


async def _fake_plan(*, run, user_input, gateway, capability_registry=None, **kwargs: object):
    del kwargs
    _fake_plan.calls.append(
        {
            "run": run,
            "user_input": user_input,
            "gateway": gateway,
            "capability_registry": capability_registry,
        }
    )
    if _fake_plan.error is not None:
        raise _fake_plan.error
    return _FakeResult()


_fake_plan.calls = []
_fake_plan.error = None


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(params={"mode": "plan", "draft": "x"}, id="plandrive1")


class TestDrivePlanMode:
    def test_successful_pipeline_marks_run_succeeded(self, run) -> None:
        _fake_plan.calls = []
        _fake_plan.error = None
        result = asyncio.run(
            drive_plan_mode(run=run, user_input="x", gateway=object(), plan=_fake_plan)
        )
        assert isinstance(result, _FakeResult)
        assert run.status == "succeeded"
        assert _fake_plan.calls[0]["user_input"] == "x"

    def test_failed_pipeline_marks_run_failed_and_propagates(self, run) -> None:
        from molexp.harness import StageExecutionError

        _fake_plan.calls = []
        _fake_plan.error = StageExecutionError("stage 'x' exploded")
        with pytest.raises(StageExecutionError):
            asyncio.run(drive_plan_mode(run=run, user_input="x", gateway=object(), plan=_fake_plan))
        assert run.status == "failed"

    def test_reentry_on_a_succeeded_run_is_allowed(self, run) -> None:
        _fake_plan.calls = []
        _fake_plan.error = None
        asyncio.run(drive_plan_mode(run=run, user_input="x", gateway=object(), plan=_fake_plan))
        asyncio.run(drive_plan_mode(run=run, user_input="x", gateway=object(), plan=_fake_plan))
        assert run.status == "succeeded"
        assert len(_fake_plan.calls) == 2
