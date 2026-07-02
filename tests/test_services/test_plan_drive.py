"""``drive_plan_mode`` — PlanMode inside the run lifecycle (honest status).

Regression: a plan Run's workspace status stayed ``pending`` forever — the
CLI and the server task both drove ``PlanMode().run(...)`` bare, so a run
whose 13 stages all completed still showed ``Pending 1 / Succeeded 0`` in
every UI status surface ("等于让我不信任所有状态徽章"). The shared driver
wraps the pipeline in ``run.start()`` so the status machine, ownership
stamp, and heartbeat apply to plan runs exactly as they do to workflow runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Workspace


class _FakeResult:
    pass


class _FakeMode:
    """Duck-typed Mode: async run(...) succeeding or raising."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[dict] = []

    async def run(self, *, run, user_input, gateway, capability_registry=None):
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


def test_successful_pipeline_marks_the_run_succeeded(run) -> None:
    mode = _FakeMode()
    result = asyncio.run(drive_plan_mode(mode, run=run, user_input="x", gateway=object()))
    assert isinstance(result, _FakeResult)
    assert run.status == "succeeded"
    assert mode.calls[0]["user_input"] == "x"


def test_failed_pipeline_marks_the_run_failed_and_propagates(run) -> None:
    from molexp.harness import StageExecutionError

    mode = _FakeMode(error=StageExecutionError("stage 'x' exploded"))
    with pytest.raises(StageExecutionError):
        asyncio.run(drive_plan_mode(mode, run=run, user_input="x", gateway=object()))
    assert run.status == "failed"


def test_reentry_on_a_succeeded_run_is_allowed(run) -> None:
    """Plan re-entry is governed by the stage ledger, not the run verbs — a
    re-run against a succeeded plan run must enter the lifecycle again (and
    end succeeded again), not refuse like `molexp run` would."""
    mode = _FakeMode()
    asyncio.run(drive_plan_mode(mode, run=run, user_input="x", gateway=object()))
    asyncio.run(drive_plan_mode(mode, run=run, user_input="x", gateway=object()))
    assert run.status == "succeeded"
    assert len(mode.calls) == 2
