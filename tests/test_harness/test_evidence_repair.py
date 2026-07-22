"""Evidence diagnosis + RepairLoop hard-stop vs. retry on capability gaps.

Mirrors ``molexp.harness.stages.evidence`` (``diagnose_failure`` /
``collect_evidence_text``) and ``molexp.harness.stages.repair_loop``
(``RepairLoop``). The one distinction under test: a missing molcrafts API
(``api_symbol_missing`` / confirmed ``SYMBOL_NOT_FOUND``) is a hard capability
gap that stops the plan, while an *invented* ``unknown_capability`` id from a
live catalog — and an environmental ``molmcp_unavailable`` miss — stay
retryable.
"""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import CapabilityGapError, StagePersistedFailureError
from molexp.harness.stages.evidence import collect_evidence_text, diagnose_failure
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


class TestDiagnoseFailure:
    def test_unknown_capability_is_retryable_not_hard_gap(self) -> None:
        """Binder inventing an id is re-bindable — do not hard-stop the plan."""
        report = {
            "passed": False,
            "violations": [
                {
                    "code": "unknown_capability",
                    "message": "BoundTask 't' references capability_id 'molpy.nope.X' which is not in the registry",
                    "path": "tasks",
                }
            ],
        }
        d = diagnose_failure(Exception("x"), json.dumps(report))
        assert not d.is_capability_gap
        assert "unknown_capability" in d.codes
        assert any("molpy.nope.X" in s for s in d.symbols)


class TestCollectEvidenceText:
    def test_confirmed_symbol_miss_promotes_to_capability_gap(self) -> None:
        d = diagnose_failure(
            Exception("x"),
            "AttributeError: 'molpy.core.box.Box' object has no attribute 'lengths'",
        )
        assert "molpy.core.box.Box.lengths" in d.symbols or any("Box" in s for s in d.symbols)

        def lookup(symbol: str) -> dict:
            return {"ok": False, "code": "SYMBOL_NOT_FOUND", "error": "gone", "ref": symbol}

        text = collect_evidence_text(d, lookup=lookup)
        assert "SYMBOL_NOT_FOUND" in text
        assert d.is_capability_gap

    def test_molmcp_unavailable_is_not_a_capability_gap(self) -> None:
        d = diagnose_failure(Exception("x"), "use molpy.core.box.Box.lengths please")
        text = collect_evidence_text(
            d,
            lookup=lambda s: {
                "ok": False,
                "code": "molmcp_unavailable",
                "error": "no collection",
                "ref": s,
            },
        )
        assert "molmcp_unavailable" in text
        # Environmental miss must not burn the plan as a capability gap.
        assert not d.is_capability_gap


class _Gen(Stage):
    name: ClassVar[str] = "gen"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx):
        self.calls += 1
        return ctx.artifact_store.put_json(
            kind="thing", obj={"n": self.calls}, created_by="gen", parent_ids=[]
        )


class _AlwaysUnknownCapability(Stage):
    name: ClassVar[str] = "val"

    async def run(self, ctx):
        report = ctx.artifact_store.put_json(
            kind="validation_report",
            obj={
                "passed": False,
                "violations": [
                    {
                        "code": "unknown_capability",
                        "message": "capability_id 'molpy.missing.Thing' not in the registry",
                    }
                ],
            },
            created_by="val",
            parent_ids=[],
        )
        raise StagePersistedFailureError(report, "validation failed")


class _AlwaysSymbolMissing(Stage):
    name: ClassVar[str] = "val"

    async def run(self, ctx):
        report = ctx.artifact_store.put_json(
            kind="validation_report",
            obj={
                "passed": False,
                "violations": [
                    {
                        "code": "api_symbol_missing",
                        "message": "molpy.core.box.Box.lengths is not available",
                    }
                ],
            },
            created_by="val",
            parent_ids=[],
        )
        raise StagePersistedFailureError(report, "validation failed")


class TestRepairLoop:
    def test_retries_unknown_capability_until_budget_exhausted(self, tmp_path) -> None:
        """Invented capability_id → re-generate (do not hard-stop on first miss)."""
        ctx = _ctx(tmp_path)
        gen = _Gen()
        loop = RepairLoop(
            name="thing",
            generate=gen,
            validators=[_AlwaysUnknownCapability()],
            feedback_kind="gen_feedback",
            attempts=3,
        )
        with pytest.raises(StagePersistedFailureError):
            asyncio.run(loop.run(ctx))
        assert gen.calls == 3  # burned the budget rebinding
        fb = ctx.artifact_store.latest_by_kind("gen_feedback")
        assert fb is not None
        assert b"unknown_capability" in ctx.artifact_store.get(fb.id)

    def test_hard_stops_on_capability_gap_without_retrying(self, tmp_path) -> None:
        ctx = _ctx(tmp_path)
        gen = _Gen()
        loop = RepairLoop(
            name="thing",
            generate=gen,
            validators=[_AlwaysSymbolMissing()],
            feedback_kind="gen_feedback",
            attempts=3,
        )
        with pytest.raises(CapabilityGapError) as ei:
            asyncio.run(loop.run(ctx))
        assert "MolpyCapabilityMissing" in str(ei.value)
        assert gen.calls == 1  # no lottery retries
        fb = ctx.artifact_store.latest_by_kind("gen_feedback")
        assert fb is not None
        assert b"api_symbol_missing" in ctx.artifact_store.get(fb.id)
