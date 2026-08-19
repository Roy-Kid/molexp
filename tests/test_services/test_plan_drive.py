"""``drive_plan_mode`` — plan bundle inside the run lifecycle (honest status)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Workspace


class _FakeResult:
    pass


class _FakePlan:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def run(
        self,
        *,
        run: object,
        user_input: str,
        gateway: object,
        capability_registry: object = None,
    ) -> _FakeResult:
        self.calls.append(
            {
                "run": run,
                "user_input": user_input,
                "gateway": gateway,
                "capability_registry": capability_registry,
            }
        )
        if self._error is not None:
            raise self._error
        return _FakeResult()


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(params={"mode": "plan", "draft": "x"}, id="plandrive1")


class TestDrivePlanMode:
    def test_successful_pipeline_marks_run_succeeded(self, run) -> None:
        plan = _FakePlan()
        result = asyncio.run(drive_plan_mode(plan, run=run, user_input="x", gateway=object()))
        assert isinstance(result, _FakeResult)
        assert run.status == "succeeded"
        assert plan.calls[0]["user_input"] == "x"

    def test_failed_pipeline_marks_run_failed_and_propagates(self, run) -> None:
        from molexp.harness import StageExecutionError

        plan = _FakePlan(error=StageExecutionError("stage 'x' exploded"))
        with pytest.raises(StageExecutionError):
            asyncio.run(drive_plan_mode(plan, run=run, user_input="x", gateway=object()))
        assert run.status == "failed"

    def test_reentry_on_a_succeeded_run_is_allowed(self, run) -> None:
        plan = _FakePlan()
        asyncio.run(drive_plan_mode(plan, run=run, user_input="x", gateway=object()))
        asyncio.run(drive_plan_mode(plan, run=run, user_input="x", gateway=object()))
        assert run.status == "succeeded"
        assert len(plan.calls) == 2
