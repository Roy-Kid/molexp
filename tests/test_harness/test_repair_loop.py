"""Unit tests for the RepairLoop stage (generate -> validate -> repair)."""

from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.stages.repair_loop import RepairLoop
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore


def _ctx(tmp_path):
    db = tmp_path / "events.sqlite"
    store = FileArtifactStore(root=tmp_path / "artifacts")
    return HarnessRunContext(
        run_id="r",
        workspace_root=tmp_path,
        artifact_store=store,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=store),
    )


class _Gen(Stage):
    name: ClassVar[str] = "gen"

    def __init__(self) -> None:
        self.calls = 0
        self.saw_feedback: list[bool] = []

    async def run(self, ctx):
        self.calls += 1
        # Record whether the previous attempt's feedback is visible to this one.
        self.saw_feedback.append(ctx.artifact_store.latest_by_kind("gen_feedback") is not None)
        return ctx.artifact_store.put_json(
            kind="thing", obj={"attempt": self.calls}, created_by="gen", parent_ids=[]
        )


class _ValidateOkOnNth(Stage):
    name: ClassVar[str] = "val"

    def __init__(self, ok_on: int) -> None:
        self.ok_on = ok_on
        self.calls = 0

    async def run(self, ctx):
        self.calls += 1
        if self.calls < self.ok_on:
            report = ctx.artifact_store.put_json(
                kind="validation_report",
                obj={"passed": False, "violations": [f"bad-{self.calls}"]},
                created_by="val",
                parent_ids=[],
            )
            raise StagePersistedFailureError(report, "validation failed")
        return ctx.artifact_store.latest_by_kind("thing")


def test_repair_loop_regenerates_with_feedback_until_valid(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    gen = _Gen()
    val = _ValidateOkOnNth(ok_on=3)
    loop = RepairLoop(
        name="thing", generate=gen, validators=[val], feedback_kind="gen_feedback", attempts=5
    )

    ref = asyncio.run(loop.run(ctx))

    assert ref.kind == "thing"  # returns the GENERATED artifact, not the report
    assert gen.calls == 3  # failed twice, succeeded on the third attempt
    # First attempt has no feedback; attempts 2 and 3 see the recorded failure.
    assert gen.saw_feedback == [False, True, True]
    # The feedback carries the validator's persisted report content.
    fb = ctx.artifact_store.latest_by_kind("gen_feedback")
    assert fb is not None
    assert b"bad-2" in ctx.artifact_store.get(fb.id)


def test_repair_loop_raises_after_exhausting_attempts(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    gen = _Gen()
    val = _ValidateOkOnNth(ok_on=99)  # never passes
    loop = RepairLoop(
        name="thing", generate=gen, validators=[val], feedback_kind="gen_feedback", attempts=3
    )

    with pytest.raises(StagePersistedFailureError):
        asyncio.run(loop.run(ctx))
    assert gen.calls == 3  # exhausted the budget, then re-raised


def test_repair_loop_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="attempts"):
        RepairLoop(name="x", generate=_Gen(), validators=[_Gen()], feedback_kind="f", attempts=0)
    with pytest.raises(ValueError, match="validator"):
        RepairLoop(name="x", generate=_Gen(), validators=[], feedback_kind="f")
