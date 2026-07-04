"""``bypass_cache`` / fresh-execution escape hatch (run-recovery bug 4).

``--rerun`` opens a new execution but the content-addressed cache may still
serve deterministic tasks — tasks with side effects would silently not re-run.
``WorkflowRuntime.execute(bypass_cache=True)`` skips cache READS (every body
actually runs) while results are still written back to the cache. The same
request survives a process boundary as a persisted marker
(``executions/<exec_id>/fresh.json`` via ``request_fresh_execution``), which
the runtime picks up when it executes that execution id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import (
    Caching,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
    fresh_requested,
    request_fresh_execution,
)
from molexp.workspace import Workspace

_COUNTERS: dict[str, int] = {}


def _bump(name: str) -> int:
    _COUNTERS[name] = _COUNTERS.get(name, 0) + 1
    return _COUNTERS[name]


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    _COUNTERS.clear()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


def _new_run(workspace: Workspace, name: str):
    project = workspace.add_project(name=f"p-{name}")
    experiment = project.add_experiment(name=f"e-{name}")
    return experiment.add_run(params={})


def _counted_workflow():
    wf = WorkflowCompiler(name="counted-fresh")

    @wf.task
    async def step(ctx: TaskContext) -> int:
        _bump("step")
        return 42

    return wf.compile()


@pytest.mark.asyncio
async def test_bypass_cache_reruns_body(workspace: Workspace) -> None:
    compiled = _counted_workflow()
    cache = Caching(store=workspace.cache.as_cache_store())

    run1 = _new_run(workspace, "warm")
    with run1.start() as ctx1:
        await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache)
    assert _COUNTERS["step"] == 1

    # Control: a plain second run is served from cache (body not re-run).
    run2 = _new_run(workspace, "cached")
    with run2.start() as ctx2:
        await WorkflowRuntime().execute(compiled, run_context=ctx2, cache=cache)
    assert _COUNTERS["step"] == 1

    # bypass_cache=True: the body MUST run again despite the warm cache.
    run3 = _new_run(workspace, "fresh")
    with run3.start() as ctx3:
        result = await WorkflowRuntime().execute(
            compiled, run_context=ctx3, cache=cache, bypass_cache=True
        )
    assert result.outputs["step"] == 42
    assert _COUNTERS["step"] == 2


@pytest.mark.asyncio
async def test_bypass_cache_still_writes_cache(workspace: Workspace) -> None:
    """Bypass skips the READ only — the fresh result still lands in the cache."""
    compiled = _counted_workflow()
    cache = Caching(store=workspace.cache.as_cache_store())

    run1 = _new_run(workspace, "seed")
    with run1.start() as ctx1:
        await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache, bypass_cache=True)
    assert _COUNTERS["step"] == 1

    # The bypassed run populated the cache: a later normal run hits it.
    run2 = _new_run(workspace, "after")
    with run2.start() as ctx2:
        await WorkflowRuntime().execute(compiled, run_context=ctx2, cache=cache)
    assert _COUNTERS["step"] == 1


@pytest.mark.asyncio
async def test_fresh_marker_triggers_bypass_across_process_boundary(
    workspace: Workspace,
) -> None:
    """A persisted ``fresh.json`` marker requests the same bypass — the channel
    the server rerun endpoint / molq worker path uses."""
    compiled = _counted_workflow()
    cache = Caching(store=workspace.cache.as_cache_store())

    run1 = _new_run(workspace, "warm")
    with run1.start() as ctx1:
        await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache)
    assert _COUNTERS["step"] == 1

    run2 = _new_run(workspace, "marked")
    run_dir = Path(str(run2.run_dir))
    execution_id = f"exec-{run2.id}"
    marker = request_fresh_execution(run_dir, execution_id)
    assert marker.exists()
    assert fresh_requested(run_dir, execution_id)

    with run2.start(execution_id=execution_id) as ctx2:
        await WorkflowRuntime().execute(
            compiled, run_context=ctx2, cache=cache, execution_id=execution_id
        )
    # No explicit kwarg — the marker alone must force the body to re-run.
    assert _COUNTERS["step"] == 2
