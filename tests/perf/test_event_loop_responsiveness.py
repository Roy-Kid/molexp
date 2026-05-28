"""Event-loop responsiveness probe (perf-hardening-01, ac-007).

A heartbeat coroutine ticking every ~5 ms runs under ``asyncio.gather``
alongside a multi-stage pipeline run backed by REAL stores
(``FileArtifactStore`` + ``SQLiteEventLog`` + ``SQLiteProvenanceStore`` on
``tmp_path``). The max observed inter-tick gap must stay < 50 ms, proving
the loop is not blocked by persistence writes.

Stdlib only — ``time.perf_counter`` + ``asyncio``, no pytest-benchmark
(per spec out-of-scope). Marked ``@pytest.mark.perf``.

Note: with real fast WAL sqlite this may already pass before the offload
is implemented — it is the regression GUARD validated for real in the
benches. The deterministic RED for ac-003 lives in
``tests/test_harness/test_stage_runner_async_offload.py``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

TICK_SECONDS = 0.005
MAX_ALLOWED_GAP_SECONDS = 0.050  # ac-007 tolerance
N_STAGES = 8


async def _heartbeat(stop: asyncio.Event, gaps: list[float]) -> None:
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(TICK_SECONDS)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


@pytest.mark.perf
@pytest.mark.asyncio
async def test_heartbeat_not_starved_during_multi_stage_run(tmp_path: Path) -> None:
    """ac-007 — co-running heartbeat stays responsive across a real run."""
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteProvenanceStore(path=db_path, artifact_store=artifacts)

    ctx = HarnessRunContext(
        run_id="run-perf",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        provenance_store=provenance,
    )

    class ChainStage(Stage):
        """Trivial stage: derives a new artifact from the previous one."""

        def __init__(self, index: int, parent_id: str | None) -> None:
            self.name = f"ChainStage{index}"
            self._index = index
            self._parent_id = parent_id

        async def run(self, ctx: HarnessRunContext):
            parents = [self._parent_id] if self._parent_id else []
            return ctx.artifact_store.put_json(
                kind="log",
                obj={"step": self._index},
                created_by=self.name,
                parent_ids=parents,
            )

    runner = StageRunner(ctx)
    gaps: list[float] = []
    stop = asyncio.Event()

    async def _drive() -> None:
        prev_id: str | None = None
        for i in range(N_STAGES):
            ref = await runner.run_stage(ChainStage(i, prev_id))
            prev_id = ref.id
        stop.set()

    hb_task = asyncio.create_task(_heartbeat(stop, gaps))
    # Give the heartbeat a few ticks before the run starts so it is
    # genuinely live; its clock resets on the last tick before _drive runs.
    await asyncio.sleep(3 * TICK_SECONDS)
    await _drive()
    await hb_task

    assert gaps, "heartbeat never ticked"
    max_gap = max(gaps)
    assert max_gap < MAX_ALLOWED_GAP_SECONDS, (
        f"event loop starved during multi-stage run: max inter-tick gap "
        f"{max_gap * 1e3:.1f} ms >= {MAX_ALLOWED_GAP_SECONDS * 1e3:.0f} ms"
    )
