"""StageRunner async-offload guard (perf-hardening-01, ac-003).

Deterministic, sqlite-latency-independent RED/GREEN test: a stub
``event_log`` / ``lineage_store`` whose ``append`` / ``add_edge`` block
the calling thread for a fixed 50 ms via ``time.sleep``. ``run_stage`` runs
under ``asyncio.gather`` alongside a heartbeat coroutine that records
``time.perf_counter`` every ~5 ms and computes the max inter-tick gap.

Because ``StageRunner`` dispatches each ``append`` / ``add_edge`` through
``asyncio.to_thread``, the blocking sleeps run on worker threads and the loop
keeps ticking — max gap stays near the 5 ms tick. A regression that put the
sync store calls back on the loop would block it for the full 50 ms of each
write, pushing the max inter-tick gap past the tolerance and failing here.

The stubs satisfy exactly what ``StageRunner`` calls: ``append`` with
``run_id`` / ``type`` / ``actor`` / ``payload`` (+ optional ``artifact_ids``)
returning a real :class:`HarnessEvent`, and ``add_edge(parent_id, child_id,
relation)``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from molexp.harness.schemas import EventType, HarnessEvent

# 50 ms blocking cost per persistence call — large vs the 5 ms tick.
BLOCKING_SECONDS = 0.05
TICK_SECONDS = 0.005
# A starved loop sees ~50 ms gaps; an offloaded loop stays near the tick.
MAX_ALLOWED_GAP_SECONDS = 0.030


class _BlockingEventLog:
    """Stub ``EventLog`` whose ``append`` blocks the calling thread 50 ms."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def append(
        self,
        run_id: str,
        type: EventType,
        actor: str,
        payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> HarnessEvent:
        time.sleep(BLOCKING_SECONDS)
        self.calls.append({"run_id": run_id, "type": type, "actor": actor})
        return HarnessEvent(
            id=uuid.uuid4().hex,
            run_id=run_id,
            seq=len(self.calls),
            type=type,
            actor=actor,
            created_at=datetime.now(tz=UTC),
            payload=payload or {},
            artifact_ids=list(artifact_ids or []),
        )

    def list_events(self, run_id: str) -> list[HarnessEvent]:  # pragma: no cover
        return []

    def get_timeline(self, run_id: str) -> list[HarnessEvent]:  # pragma: no cover
        return []


class _BlockingArtifactLineageStore:
    """Stub ``ArtifactLineageStore`` whose ``add_edge`` blocks the thread 50 ms."""

    def __init__(self) -> None:
        self.edges: list[tuple[str, str, str]] = []

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
        *,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> None:
        time.sleep(BLOCKING_SECONDS)
        self.edges.append((parent_id, child_id, relation))


async def _heartbeat(stop: asyncio.Event, gaps: list[float]) -> None:
    """Tick every ~5 ms; record each observed inter-tick gap."""
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(TICK_SECONDS)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


@pytest.mark.asyncio
async def test_run_stage_does_not_starve_heartbeat(tmp_path: Path) -> None:
    """ac-003 — persistence writes must not block the event loop.

    RED today: ``StageRunner`` calls the blocking ``append`` synchronously
    on the loop, so the heartbeat is frozen for the full 50 ms of each call
    and the max inter-tick gap exceeds the 30 ms tolerance.
    """
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    event_log = _BlockingEventLog()
    provenance = _BlockingArtifactLineageStore()

    ctx = HarnessRunContext(
        run_id="run-offload",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=event_log,  # type: ignore[arg-type]
        lineage_store=provenance,  # type: ignore[arg-type]
    )

    class NoopStage(Stage):
        name = "NoopStage"

        async def run(self, ctx: HarnessRunContext):
            return ctx.artifact_store.put_json(
                kind="log",
                obj={"note": "hello"},
                created_by="NoopStage",
                parent_ids=[],
            )

    runner = StageRunner(ctx)
    gaps: list[float] = []
    stop = asyncio.Event()

    hb_task = asyncio.create_task(_heartbeat(stop, gaps))
    # Let the heartbeat take its first tick so it is genuinely running
    # before the stage's persistence writes begin. After this yield the
    # heartbeat's clock is freshly reset; any starvation during run_stage
    # then shows up as a single oversized inter-tick gap.
    await asyncio.sleep(3 * TICK_SECONDS)

    await runner.run_stage(NoopStage())
    stop.set()
    await hb_task

    # The stub stores must actually have been exercised.
    assert event_log.calls, "event_log.append was never called"

    assert gaps, "heartbeat never ticked"
    max_gap = max(gaps)
    assert max_gap < MAX_ALLOWED_GAP_SECONDS, (
        f"event loop starved: max inter-tick gap {max_gap * 1e3:.1f} ms "
        f">= {MAX_ALLOWED_GAP_SECONDS * 1e3:.0f} ms tolerance — persistence "
        f"writes are still blocking the loop (expected after asyncio.to_thread offload)"
    )
