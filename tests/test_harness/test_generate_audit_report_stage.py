"""Tests for the ``GenerateAuditReport`` stage (spec ``harness-run-mode-01-substrate``, T08).

Contract under test (RED — the stage does not exist yet):
- ``name == "generate_audit_report"``; wraps the existing pure function
  ``generate_audit_report(run_id=ctx.run_id, event_log=ctx.event_log,
  artifact_store=ctx.artifact_store, lineage_store=ctx.lineage_store)``
  (``src/molexp/harness/audit/generate.py``);
- persists the resulting ``AuditReport`` as an ``"audit_report"`` artifact
  with ``created_by="GenerateAuditReport"`` and ``parent_ids=[]`` (a
  run-level summary derives from the whole run, not one artifact);
- the persisted JSON round-trips through ``molexp.harness.AuditReport``
  with ``run_id == ctx.run_id``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-audit",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_some_artifacts(store) -> None:
    """Make the store non-trivial so the report summarizes real content."""
    plan_ref = store.put_json(
        kind="user_plan",
        obj={"text": "estimate the diffusion coefficient"},
        created_by="seed",
        parent_ids=[],
    )
    store.put_json(
        kind="experiment_report",
        obj={"title": "demo", "objective": "demo objective"},
        created_by="seed",
        parent_ids=[plan_ref.id],
    )


def test_name_and_subclass() -> None:
    from molexp.harness import GenerateAuditReport, Stage

    assert GenerateAuditReport.name == "generate_audit_report"
    assert issubclass(GenerateAuditReport, Stage)


def test_persists_audit_report_artifact(ctx) -> None:
    from molexp.harness import GenerateAuditReport

    _seed_some_artifacts(ctx.artifact_store)
    ref = asyncio.run(GenerateAuditReport().run(ctx))

    assert ref.kind == "audit_report"
    assert ref.parent_ids == []
    assert ref.created_by == "GenerateAuditReport"


def test_audit_report_round_trips_with_run_id(ctx) -> None:
    from molexp.harness import AuditReport, GenerateAuditReport

    _seed_some_artifacts(ctx.artifact_store)
    ref = asyncio.run(GenerateAuditReport().run(ctx))

    report = AuditReport.model_validate(json.loads(ctx.artifact_store.get(ref.id)))
    assert report.run_id == ctx.run_id
