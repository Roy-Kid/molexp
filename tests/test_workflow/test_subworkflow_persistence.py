"""SubWorkflow nested execution must not clobber the outer ``workflow.json``.

An inner run inherits the OUTER ``run_context`` (same run dir + active
execution id); if it persisted, it would rewrite
``executions/<exec_id>/workflow.json`` with the INNER spec's document —
losing the outer graph/statuses, polluting resume seeds, and racing under
``wf.parallel`` fan-out. Invariant pinned here: after a run containing
SubWorkflows (including parallel fan-out, and after an inner failure),
``executions/<exec_id>/workflow.json`` describes the OUTER graph only, with
correct statuses, and resume seeding reads only outer-node outputs.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from molexp.workflow import (
    SubWorkflow,
    WorkflowCompiler,
    WorkflowRuntime,
    read_node_outputs,
)
from molexp.workspace import Workspace

INNER_TASKS = {"load", "normalize", "scale"}


def _new_run(tmp_path: pathlib.Path, params: dict | None = None):
    ws = Workspace(tmp_path / "lab")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params=params or {})


def _build_inner() -> WorkflowCompiler:
    """A 3-task inner chain: load → normalize → scale (terminal = 1.75)."""
    wf = WorkflowCompiler(name="inner-multi")

    @wf.task
    async def load() -> list[float]:
        return [2.0, 4.0, 8.0]

    @wf.task(depends_on=["load"])
    async def normalize(values: list[float]) -> list[float]:
        top = max(values)
        return [x / top for x in values]

    @wf.task(depends_on=["normalize"])
    async def scale(values: list[float]) -> float:
        return sum(values)

    return wf


def _build_failing_inner() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="inner-fail")

    @wf.task
    async def boom() -> None:
        raise ValueError("inner exploded")

    return wf


def _load_doc(run, execution_id: str) -> dict:
    path = pathlib.Path(run.run_dir) / "executions" / execution_id / "workflow.json"
    return json.loads(path.read_text())


def _task_statuses(doc: dict) -> dict[str, str]:
    return {t["task_id"]: t["status"] for t in doc["task_configs"]}


# ── success: parent document is the OUTER graph, fully statused ──────────────


@pytest.mark.asyncio
async def test_parent_workflow_json_describes_outer_graph_after_success(
    tmp_path: pathlib.Path,
) -> None:
    outer = (
        WorkflowCompiler(name="outer-doc").add(SubWorkflow(_build_inner()), name="sub").compile()
    )
    run = _new_run(tmp_path)
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(outer, run_context=ctx)

    assert result.status == "completed"
    doc = _load_doc(run, result.execution_id)
    # OUTER graph only — never the inner spec's document.
    assert doc["workflow_name"] == "outer-doc"
    statuses = _task_statuses(doc)
    assert set(statuses) == {"sub"}
    assert not (set(statuses) & INNER_TASKS)
    assert statuses["sub"] == "completed"
    assert doc["status"] == "completed"

    # Resume seeding reads only outer-node outputs.
    seeds = read_node_outputs(run.run_dir, result.execution_id)
    assert set(seeds) == {"sub"}
    assert seeds["sub"] == pytest.approx(1.75)


# ── inner failure: outer document survives and marks the node failed ─────────


@pytest.mark.asyncio
async def test_parent_workflow_json_intact_after_inner_failure(
    tmp_path: pathlib.Path,
) -> None:
    outer = (
        WorkflowCompiler(name="outer-doc-fail")
        .add(SubWorkflow(_build_failing_inner()), name="sub")
        .compile()
    )
    run = _new_run(tmp_path)
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(outer, run_context=ctx)

    assert result.status == "failed"
    doc = _load_doc(run, result.execution_id)
    assert doc["workflow_name"] == "outer-doc-fail"
    statuses = _task_statuses(doc)
    assert set(statuses) == {"sub"}  # outer graph only, even mid-failure
    assert statuses["sub"] == "failed"
    assert doc["status"] == "failed"


# ── parallel fan-out of subworkflows does not corrupt the document ───────────


@pytest.mark.asyncio
async def test_parallel_subworkflow_fanout_does_not_corrupt_parent_doc(
    tmp_path: pathlib.Path,
) -> None:
    wf = WorkflowCompiler(name="outer-doc-parallel", entry="enumerate")

    @wf.task
    async def enumerate() -> list[int]:
        return [0, 1, 2, 3]

    wf.add(SubWorkflow(_build_inner()), name="sub")

    @wf.task
    async def collect(values: list[float]) -> list[float]:
        return list(values)

    wf.parallel(map_over="enumerate", body="sub", join="collect", max_concurrency=4)

    run = _new_run(tmp_path)
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

    assert result.status == "completed"
    doc = _load_doc(run, result.execution_id)
    assert doc["workflow_name"] == "outer-doc-parallel"
    statuses = _task_statuses(doc)
    assert set(statuses) == {"enumerate", "sub", "collect"}
    assert not (set(statuses) & INNER_TASKS)
    assert statuses == {"enumerate": "completed", "sub": "completed", "collect": "completed"}
    assert doc["status"] == "completed"

    seeds = read_node_outputs(run.run_dir, result.execution_id)
    assert set(seeds) == {"enumerate", "sub", "collect"}
    assert seeds["collect"] == pytest.approx([1.75, 1.75, 1.75, 1.75])
