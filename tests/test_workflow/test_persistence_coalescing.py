"""Coalesced ``workflow.json`` writes — O(N) bytes, mandatory terminal flush.

Every per-task status transition used to rewrite the entire
``executions/<exec_id>/workflow.json`` document synchronously: a 1000-element
``wf.parallel`` meant O(N²) bytes written. The coalescing writer
(:mod:`molexp.workflow._pydantic_graph.persistence`) keeps the authoritative
document in memory during an execution, marks it dirty per transition, and
flushes at a bounded staleness (``WORKFLOW_JSON_MAX_STALENESS_S`` — a
performance knob, never a correctness gate).

Pinned here:

* **Write-count bound** — an N-element parallel execution performs far fewer
  than the per-transition writer's ≥N full-document writes, while the FINAL
  document is byte-equivalent to what per-transition writing leaves.
* **Mandatory synchronous flushes** — a task failure and the execution
  terminal state always land on disk even with the coalescer mid-interval.
* **Bounded staleness** — a dirty document reaches disk within the staleness
  window without any further marks.
* **No-persist stays zero-cost** — ``persist=False`` (SubWorkflow inner runs)
  performs zero document writes.

The flusher is a daemon ``threading.Timer``; engine coordination never waits
on it (see ``test_pg_lowering.py::test_no_timing_constants_for_coordination``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime
from molexp.workflow._pydantic_graph import persistence

if TYPE_CHECKING:
    from molexp._typing import JSONValue

N = 60

_PINNED_NOW = "2026-01-01T00:00:00"


def _build_parallel_workflow(n: int):
    """N-element fan-out: emit → double (xN) → total."""
    wf = WorkflowCompiler(name="coalesce-fanout", entry="emit")

    @wf.task
    async def emit(ctx: TaskContext) -> list[int]:
        return list(range(n))

    @wf.task
    async def double(element: int) -> int:
        return element * 2

    @wf.task
    async def total(results: list[int]) -> int:
        return sum(results)

    wf.parallel(map_over="emit", body="double", join="total", max_concurrency=8)
    return wf.compile()


def _doc(run_dir: Path, execution_id: str) -> dict[str, JSONValue]:
    return json.loads((run_dir / "executions" / execution_id / "workflow.json").read_text())


def _task_statuses(doc: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return {t["task_id"]: t["status"] for t in doc["task_configs"] if isinstance(t, dict)}


@pytest.fixture
def count_writes(monkeypatch):
    """Count every full-document serialization through the real writer."""
    counter = {"n": 0}
    real = persistence.atomic_write_json

    def counting(path, data) -> None:
        counter["n"] += 1
        real(path, data)

    monkeypatch.setattr(persistence, "atomic_write_json", counting)
    return counter


# ── write-count bound + final-document equivalence ───────────────────────────


@pytest.mark.asyncio
async def test_parallel_write_count_bounded_and_final_document_byte_equivalent(
    tmp_path, monkeypatch, count_writes
) -> None:
    """Coalescing cuts full-document writes from ≥N to ≤ N/2 + constant while
    the final on-disk document is byte-equivalent to per-transition writing."""
    compiled = _build_parallel_workflow(N)
    # Pin timestamps so the two modes produce comparable documents.
    monkeypatch.setattr(persistence, "_now", lambda: _PINNED_NOW)

    # Reference: staleness 0 == every mark flushes synchronously — exactly the
    # old per-transition writer's behavior and write count.
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 0.0)
    sync_dir = tmp_path / "sync"
    result = await WorkflowRuntime().execute(compiled, run_dir=sync_dir, execution_id="exec-1")
    assert result.status == "completed"
    sync_writes = count_writes["n"]
    sync_bytes = (sync_dir / "executions" / "exec-1" / "workflow.json").read_bytes()

    # Coalesced (default staleness): same execution, bounded write count.
    count_writes["n"] = 0
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 0.2)
    coal_dir = tmp_path / "coalesced"
    result = await WorkflowRuntime().execute(compiled, run_dir=coal_dir, execution_id="exec-1")
    assert result.status == "completed"
    coalesced_writes = count_writes["n"]
    coalesced_bytes = (coal_dir / "executions" / "exec-1" / "workflow.json").read_bytes()

    # Per-transition writing is O(N) full-document writes (≥ one per element).
    assert sync_writes >= N, f"expected >= {N} per-transition writes, saw {sync_writes}"
    # Coalesced: initial + a few staleness flushes + the terminal flush.
    bound = N // 2 + 5
    assert coalesced_writes <= bound, (
        f"coalescing wrote the document {coalesced_writes} times for an "
        f"{N}-element parallel execution; expected <= {bound}"
    )
    assert coalesced_writes < sync_writes
    # Identical final document content — the format did not change.
    assert coalesced_bytes == sync_bytes


@pytest.mark.asyncio
async def test_result_outputs_identical_under_coalescing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 3600.0)
    compiled = _build_parallel_workflow(5)
    result = await WorkflowRuntime().execute(compiled, run_dir=tmp_path, execution_id="exec-r")
    assert result.outputs["double"] == [0, 2, 4, 6, 8]
    assert result.outputs["total"] == 20
    doc = _doc(tmp_path, "exec-r")
    statuses = _task_statuses(doc)
    assert statuses == {"emit": "completed", "double": "completed", "total": "completed"}


# ── mandatory synchronous flushes ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_failure_lands_on_disk_mid_interval(tmp_path, monkeypatch) -> None:
    """A failing task + the failed terminal state hit disk synchronously even
    when the coalescer's staleness window would not elapse for an hour."""
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 3600.0)
    wf = WorkflowCompiler(name="coalesce-failure")

    @wf.task
    async def ok(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["ok"])
    async def boom(ctx: TaskContext) -> int:
        raise RuntimeError("kaboom")

    result = await WorkflowRuntime().execute(wf.compile(), run_dir=tmp_path, execution_id="exec-f")
    assert result.status == "failed"
    doc = _doc(tmp_path, "exec-f")
    assert doc["status"] == "failed"
    assert doc["finished_at"] is not None
    statuses = _task_statuses(doc)
    assert statuses["boom"] == "failed"
    # The coalesced-but-unflushed 'ok' completion rode the terminal flush.
    assert statuses["ok"] == "completed"


@pytest.mark.asyncio
async def test_execution_success_lands_on_disk_mid_interval(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 3600.0)
    wf = WorkflowCompiler(name="coalesce-success")

    @wf.task
    async def a(ctx: TaskContext) -> int:
        return 41

    @wf.task(depends_on=["a"])
    async def b(value: int) -> int:
        return value + 1

    result = await WorkflowRuntime().execute(wf.compile(), run_dir=tmp_path, execution_id="exec-s")
    assert result.status == "completed"
    doc = _doc(tmp_path, "exec-s")
    assert doc["status"] == "completed"
    assert _task_statuses(doc) == {"a": "completed", "b": "completed"}
    by_name = {t["task_id"]: t for t in doc["task_configs"]}
    assert by_name["b"]["outputs"] == 42


# ── bounded staleness: the flusher writes without further marks ──────────────


def test_dirty_document_flushes_within_bounded_staleness(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 0.05)
    compiled = _build_parallel_workflow(2)
    persistence.open_execution_document(tmp_path, "exec-d", compiled=compiled)
    try:
        persistence.mark_task_status(tmp_path, "exec-d", "emit", "running")
        # Not yet flushed — the mark only made the in-memory document dirty.
        assert _task_statuses(_doc(tmp_path, "exec-d"))["emit"] == "pending"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _task_statuses(_doc(tmp_path, "exec-d"))["emit"] == "running":
                break
            time.sleep(0.02)
        assert _task_statuses(_doc(tmp_path, "exec-d"))["emit"] == "running", (
            "dirty document did not reach disk within the bounded staleness window"
        )
    finally:
        persistence.close_execution_document(tmp_path, "exec-d")


def test_close_flushes_pending_state_and_is_idempotent(tmp_path, monkeypatch) -> None:
    """The finally-path close (engine raised, no terminal mark) still lands
    the last in-memory state on disk."""
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 3600.0)
    compiled = _build_parallel_workflow(2)
    persistence.open_execution_document(tmp_path, "exec-c", compiled=compiled)
    persistence.mark_task_status(tmp_path, "exec-c", "emit", "running")
    assert _task_statuses(_doc(tmp_path, "exec-c"))["emit"] == "pending"
    persistence.close_execution_document(tmp_path, "exec-c")
    assert _task_statuses(_doc(tmp_path, "exec-c"))["emit"] == "running"
    # Idempotent + None-tolerant (the runtime calls it from finally blindly).
    persistence.close_execution_document(tmp_path, "exec-c")
    persistence.close_execution_document(None, None)


# ── standalone (unmanaged) callers keep synchronous semantics ────────────────


def test_unmanaged_mark_task_status_writes_synchronously(tmp_path, monkeypatch) -> None:
    """Without an opened writer (e.g. the workspace run_result_fallback drift
    guard) mark_task_status keeps the legacy synchronous read-modify-write."""
    monkeypatch.setattr(persistence, "WORKFLOW_JSON_MAX_STALENESS_S", 3600.0)
    persistence.write_initial_workflow_json(tmp_path, "exec-u")
    wf_path = tmp_path / "executions" / "exec-u" / "workflow.json"
    document = json.loads(wf_path.read_text())
    document["task_configs"] = [{"task_id": "train", "status": "pending"}]
    wf_path.write_text(json.dumps(document))
    persistence.mark_task_status(
        tmp_path, "exec-u", "train", "completed", output={"loss": 0.5}, snapshot_key="k"
    )
    doc = _doc(tmp_path, "exec-u")
    assert doc["task_configs"][0]["status"] == "completed"
    assert doc["task_configs"][0]["outputs"] == {"loss": 0.5}


# ── persist=False stays zero-cost ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_persist_execution_performs_zero_document_writes(tmp_path, count_writes) -> None:
    compiled = _build_parallel_workflow(3)
    result = await WorkflowRuntime().execute(
        compiled, run_dir=tmp_path, execution_id="exec-n", persist=False
    )
    assert result.status == "completed"
    assert count_writes["n"] == 0
    assert not (tmp_path / "executions").exists()
