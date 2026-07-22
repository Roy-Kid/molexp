"""Resume-seed integrity — persisted outputs are verified before seeding.

A resumed execution seeds completed-node outputs from the prior attempt's
``executions/<exec_id>/workflow.json``. Those persisted values are only
trustworthy when (a) they were produced by the SAME task code + config
(verified via the persisted ``snapshot_key`` vs the live recomputed
``TaskSnapshot.key``) and (b) they did not go through the lossy ``_jsonable``
observability rendering (``outputs_lossy`` flag). This module owns the
verification contract in ``_engine/persistence.py`` (``filter_resume_seeds`` /
``read_node_outputs``) plus the ``execute(seed_outputs=…)`` fail-fast gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workflow import (
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
    read_node_outputs,
)
from molexp.workflow._engine.persistence import filter_resume_seeds

# ── module-level per-task execution counters ─────────────────────────────────
_COUNTERS: dict[str, int] = {}


def _bump(name: str) -> int:
    _COUNTERS[name] = _COUNTERS.get(name, 0) + 1
    return _COUNTERS[name]


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    _COUNTERS.clear()


def _compiled_returning(value: object):
    """A 1-task workflow whose body counts invocations and returns *value*.

    The returned value is baked into the body source, so two different values
    produce two different ``code_hash``es (⇒ different snapshot keys for the
    same task name) — the "task code changed between attempts" shape.
    """
    wf = WorkflowCompiler(name="resume-seed")

    if value == "old":

        @wf.task
        async def step(ctx: TaskContext) -> str:
            _bump("step")
            return "old"

    else:

        @wf.task
        async def step(ctx: TaskContext) -> str:
            _bump("step")
            return "new"

    return wf.compile()


def _wf_json_path(run_dir: Path, execution_id: str) -> Path:
    return run_dir / "executions" / execution_id / "workflow.json"


class _Opaque:
    """A non-JSON-safe output (str-ified by the lossy ``_jsonable`` path)."""


class TestResumeSeedIntegrity:
    @pytest.mark.asyncio
    async def test_completed_node_persists_snapshot_key(self, tmp_path: Path) -> None:
        """A completed node records its ``snapshot_key``; a JSON-safe output is
        full-fidelity (no ``outputs_lossy`` flag)."""
        compiled = _compiled_returning("old")
        result = await WorkflowRuntime().execute(compiled, run_dir=tmp_path)
        assert result.status == "succeeded"
        doc = json.loads(_wf_json_path(tmp_path, result.execution_id).read_text())
        (record,) = [t for t in doc["task_configs"] if t["task_id"] == "step"]
        assert record["status"] == "completed"
        assert record["snapshot_key"] == compiled.snapshots["step"].key
        assert "outputs_lossy" not in record

    @pytest.mark.asyncio
    async def test_intact_seed_skips_body(self, tmp_path: Path) -> None:
        """An intact seed (same code, full fidelity) skips the body on resume."""
        compiled = _compiled_returning("old")
        r1 = await WorkflowRuntime().execute(compiled, run_dir=tmp_path)
        assert _COUNTERS["step"] == 1

        seeds = read_node_outputs(tmp_path, r1.execution_id)
        assert seeds == {"step": "old"}

        r2 = await WorkflowRuntime().execute(
            compiled, run_dir=tmp_path, execution_id=r1.execution_id, seed_outputs=seeds
        )
        assert r2.status == "succeeded"
        assert r2.outputs["step"] == "old"
        assert _COUNTERS["step"] == 1  # body NOT rerun — seed verified intact

    @pytest.mark.asyncio
    async def test_changed_code_seed_dropped_and_recomputed(self, tmp_path: Path) -> None:
        """A seed whose task code changed between attempts (different snapshot
        key) is dropped and the node recomputed with the new value."""
        v1 = _compiled_returning("old")
        r1 = await WorkflowRuntime().execute(v1, run_dir=tmp_path)
        assert _COUNTERS["step"] == 1
        seeds = read_node_outputs(tmp_path, r1.execution_id)
        assert seeds == {"step": "old"}

        v2 = _compiled_returning("new")
        assert v2.snapshots["step"].key != v1.snapshots["step"].key
        r2 = await WorkflowRuntime().execute(
            v2, run_dir=tmp_path, execution_id=r1.execution_id, seed_outputs=seeds
        )
        assert r2.status == "succeeded"
        assert r2.outputs["step"] == "new"  # NOT the stale "old"
        assert _COUNTERS["step"] == 2  # body reran

    @pytest.mark.asyncio
    async def test_lossy_output_flagged_and_never_seeded(self, tmp_path: Path) -> None:
        """A lossy output is flagged, refused by ``read_node_outputs``, dropped
        by the engine gate even when force-fed, and recomputed on resume."""
        wf = WorkflowCompiler(name="lossy")

        @wf.task
        async def step(ctx: TaskContext) -> object:
            _bump("step")
            return _Opaque()

        compiled = wf.compile()
        r1 = await WorkflowRuntime().execute(compiled, run_dir=tmp_path)
        assert r1.status == "succeeded"
        assert _COUNTERS["step"] == 1

        doc = json.loads(_wf_json_path(tmp_path, r1.execution_id).read_text())
        (record,) = [t for t in doc["task_configs"] if t["task_id"] == "step"]
        assert record["outputs_lossy"] is True

        # read_node_outputs refuses to offer the truncated value as a seed…
        assert read_node_outputs(tmp_path, r1.execution_id) == {}

        # …and even a force-fed seed is dropped by the engine-side gate.
        forced = {"step": record["outputs"]}
        kept = filter_resume_seeds(tmp_path, r1.execution_id, forced, compiled.snapshots)
        assert kept == {}

        r2 = await WorkflowRuntime().execute(
            compiled, run_dir=tmp_path, execution_id=r1.execution_id, seed_outputs=forced
        )
        assert r2.status == "succeeded"
        assert _COUNTERS["step"] == 2  # recomputed, not seeded

    @pytest.mark.asyncio
    async def test_seed_without_persisted_snapshot_key_dropped(self, tmp_path: Path) -> None:
        """A pre-upgrade document (no ``snapshot_key``) cannot be verified, so
        its seed is dropped and the node recomputed (backward compatible)."""
        compiled = _compiled_returning("old")
        r1 = await WorkflowRuntime().execute(compiled, run_dir=tmp_path)
        assert _COUNTERS["step"] == 1

        # Simulate a workflow.json written before snapshot keys were persisted.
        wf_path = _wf_json_path(tmp_path, r1.execution_id)
        doc = json.loads(wf_path.read_text())
        for task in doc["task_configs"]:
            task.pop("snapshot_key", None)
        wf_path.write_text(json.dumps(doc))

        seeds = read_node_outputs(tmp_path, r1.execution_id)
        assert seeds == {"step": "old"}  # the value is still offered…

        r2 = await WorkflowRuntime().execute(
            compiled, run_dir=tmp_path, execution_id=r1.execution_id, seed_outputs=seeds
        )
        assert r2.status == "succeeded"
        assert _COUNTERS["step"] == 2  # …but cannot be verified ⇒ recomputed

    @pytest.mark.asyncio
    async def test_seeds_without_prior_document_pass_through(self, tmp_path: Path) -> None:
        """With no prior ``workflow.json`` there is nothing to verify against, so
        programmatic seeds (e.g. from ``WorkflowResult.outputs``) are honored."""
        compiled = _compiled_returning("old")
        result = await WorkflowRuntime().execute(
            compiled, run_dir=tmp_path, seed_outputs={"step": "from-memory"}
        )
        assert result.status == "succeeded"
        assert result.outputs["step"] == "from-memory"
        assert _COUNTERS.get("step", 0) == 0  # seeded, body skipped

    @pytest.mark.asyncio
    async def test_unknown_seed_name_fails_fast(self, tmp_path: Path) -> None:
        """An unknown seed name raises ``ValueError`` before any IO."""
        compiled = _compiled_returning("old")
        with pytest.raises(ValueError, match="unknown task name"):
            await WorkflowRuntime().execute(compiled, run_dir=tmp_path, seed_outputs={"nope": 1})
