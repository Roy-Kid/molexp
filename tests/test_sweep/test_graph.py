"""Tests for :mod:`molexp.sweep` — sweep-level parallelism.

Phase 1 scope (see ``docs/spec/unified-pydantic-graph-dispatch.md`` §6):
the sweep graph fans out over ``(mol_run, experiment)`` replicas and
bounds concurrency with a ``jobs`` semaphore.  Backend selection is
*not* part of Phase 1 — every replica executes in-process via
``experiment.workflow.execute(...)``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from molexp.sweep import SweepReplica, run_sweep


class _FakeWorkflow:
    """Minimal stand-in for :class:`WorkflowSpec`; records calls and sleeps."""

    def __init__(self, *, delay: float = 0.0, fail: bool = False, tag: str = "") -> None:
        self.delay = delay
        self.fail = fail
        self.tag = tag
        self.calls: list[tuple[Any, Any]] = []

    async def execute(self, run: Any = None, profile_config: Any = None, **_: Any) -> str:
        self.calls.append((run, profile_config))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"boom-{self.tag}")
        return f"done-{self.tag}"


@dataclass
class _FakeExperiment:
    id: str
    workflow: _FakeWorkflow


@dataclass
class _FakeRun:
    id: str


def _make_replica(rid: str, *, delay: float = 0.0, fail: bool = False) -> SweepReplica:
    return SweepReplica(
        mol_run=_FakeRun(id=rid),
        experiment=_FakeExperiment(id=rid, workflow=_FakeWorkflow(delay=delay, fail=fail, tag=rid)),
    )


@pytest.mark.asyncio
class TestRunSweep:
    async def test_empty_replicas_returns_empty_state(self) -> None:
        state = await run_sweep([], jobs=1)
        assert state.outputs == {}
        assert state.failures == {}

    async def test_single_replica_executes(self) -> None:
        rep = _make_replica("r1")
        state = await run_sweep([rep], jobs=1)
        assert state.outputs == {"r1": "done-r1"}
        assert state.failures == {}

    async def test_jobs_1_is_sequential(self) -> None:
        """Three 0.1s replicas with ``jobs=1`` take ~0.3s (serialized)."""
        delay = 0.1
        reps = [_make_replica(f"r{i}", delay=delay) for i in range(3)]
        t0 = time.perf_counter()
        state = await run_sweep(reps, jobs=1)
        wall = time.perf_counter() - t0
        assert len(state.outputs) == 3
        # 10% slack for scheduler jitter
        assert wall >= 3 * delay * 0.9, f"expected sequential ~{3*delay}s, got {wall:.3f}s"

    async def test_jobs_N_is_parallel(self) -> None:
        """Three 0.2s replicas with ``jobs=3`` take ~0.2s (all concurrent)."""
        delay = 0.2
        reps = [_make_replica(f"r{i}", delay=delay) for i in range(3)]
        t0 = time.perf_counter()
        state = await run_sweep(reps, jobs=3)
        wall = time.perf_counter() - t0
        assert len(state.outputs) == 3
        assert wall < 2 * delay, f"expected parallel ~{delay}s, got {wall:.3f}s"

    async def test_semaphore_bounds_concurrency(self) -> None:
        """Four 0.2s replicas with ``jobs=2`` take ~0.4s (two waves)."""
        delay = 0.2
        reps = [_make_replica(f"r{i}", delay=delay) for i in range(4)]
        t0 = time.perf_counter()
        state = await run_sweep(reps, jobs=2)
        wall = time.perf_counter() - t0
        assert len(state.outputs) == 4
        assert wall >= 2 * delay * 0.9, f"expected two waves ~{2*delay}s, got {wall:.3f}s"
        assert wall < 3 * delay, f"expected < {3*delay}s, got {wall:.3f}s"

    async def test_failure_isolated_does_not_block_others(self) -> None:
        """One failing replica is recorded in ``failures`` without blocking others."""
        reps = [
            _make_replica("good1"),
            _make_replica("bad", fail=True),
            _make_replica("good2"),
        ]
        state = await run_sweep(reps, jobs=3)
        assert set(state.outputs.keys()) == {"good1", "good2"}
        assert set(state.failures.keys()) == {"bad"}
        assert "boom-bad" in state.failures["bad"]

    async def test_profile_config_is_forwarded(self) -> None:
        """``profile_config`` reaches each replica's ``workflow.execute``."""
        from molexp.config import ProfileConfig

        cfg = ProfileConfig({"epochs": 7}, name="smoke")
        reps = [_make_replica(f"r{i}") for i in range(2)]
        await run_sweep(reps, profile_config=cfg, jobs=2)
        for rep in reps:
            calls = rep.experiment.workflow.calls
            assert len(calls) == 1
            _, passed_cfg = calls[0]
            assert passed_cfg is cfg

    async def test_jobs_defaults_to_1(self) -> None:
        """Without explicit jobs, concurrency is 1 (sequential)."""
        delay = 0.05
        reps = [_make_replica(f"r{i}", delay=delay) for i in range(3)]
        t0 = time.perf_counter()
        await run_sweep(reps)
        wall = time.perf_counter() - t0
        assert wall >= 3 * delay * 0.9

    async def test_jobs_clamped_to_minimum_1(self) -> None:
        """Non-positive ``jobs`` is clamped up to 1 (don't deadlock on Semaphore(0))."""
        reps = [_make_replica("r1")]
        state = await run_sweep(reps, jobs=0)
        assert state.outputs == {"r1": "done-r1"}
