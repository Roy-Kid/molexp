"""``Mode`` orchestration tests.

``Mode`` runs a declared list of ``Stage`` objects eagerly — one at a time
through the shared audit bracket — against a ``workspace.Run``, returning a
frozen ``ModeResult``. Behaviours covered here: happy path, verified caching
(identical re-run skips), resume from the failed stage, and the completion
ledger's self-describing shape + its three "verified, never stale" recompute
conditions. (The rejected-artifact eviction facet lives in
``test_ledger_eviction.py``.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import ModeResult, PlanArtifactRef
from molexp.workspace import Workspace

# ───────────────────────────────────────────────────────── fixtures / stubs


@pytest.fixture()
def run(tmp_path: Path):
    """A fresh, materialized ``workspace.Run`` under a tmp workspace."""
    ws = Workspace(tmp_path / "lab", name="mode-lab")
    ws.materialize()
    project = ws.add_project("demo")
    exp = project.add_experiment("train")
    return exp.add_run(params={"seed": 0})


class CountingStage(Stage):
    """Stub stage that bumps a shared in-process counter on every invocation."""

    def __init__(self, name: str, counter: dict[str, int], kind: str = "log") -> None:
        self.name = name  # instance-level override of the ClassVar
        self._counter = counter
        self._kind = kind

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        self._counter[self.name] = self._counter.get(self.name, 0) + 1
        return ctx.artifact_store.put_json(
            kind=self._kind,
            obj={"stage": self.name, "calls": self._counter[self.name]},
            created_by=self.name,
            parent_ids=[],
        )


class FailOnceStage(Stage):
    """Stub stage that raises on its first invocation, succeeds thereafter."""

    def __init__(self, name: str, counter: dict[str, int]) -> None:
        self.name = name
        self._counter = counter

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        self._counter[self.name] = self._counter.get(self.name, 0) + 1
        if self._counter[self.name] == 1:
            raise RuntimeError(f"{self.name} boom (first attempt)")
        return ctx.artifact_store.put_json(
            kind="log",
            obj={"stage": self.name, "calls": self._counter[self.name]},
            created_by=self.name,
            parent_ids=[],
        )


def _make_mode(stage_factory):
    """Build a concrete Mode subclass whose ``stages`` returns ``stage_factory()``."""
    from molexp.harness.mode import Mode

    class _DemoMode(Mode):
        name = "demo"

        def stages(self, user_input):
            return stage_factory(user_input)

    return _DemoMode()


def _ledger_path(run) -> Path:
    return next((Path(run.run_dir) / ".mode_ledger").glob("*.json"))


class TestMode:
    """Eager stage execution + the verified completion ledger."""

    def test_run_executes_all_stages_and_returns_mode_result(self, run) -> None:
        """Declared stages each run once; ModeResult carries a ref per stage in
        declared order and the final artifact is the last stage's product."""
        counter: dict[str, int] = {}
        mode = _make_mode(
            lambda _ui: [
                CountingStage("StageA", counter, kind="user_plan"),
                CountingStage("StageB", counter, kind="experiment_report"),
                CountingStage("StageC", counter, kind="final_report"),
            ]
        )

        result = asyncio.run(mode.run(run=run, user_input={"goal": "x"}))

        assert isinstance(result, ModeResult)
        assert result.mode_name == "demo"
        assert result.run_id == run.id
        assert counter == {"StageA": 1, "StageB": 1, "StageC": 1}
        assert [a.kind for a in result.stage_artifacts] == [
            "user_plan",
            "experiment_report",
            "final_report",
        ]
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "final_report"

    def test_run_rejects_empty_stage_list(self, run) -> None:
        """An empty stage list is rejected."""
        mode = _make_mode(lambda _ui: [])
        with pytest.raises((ValueError, StageExecutionError)):
            asyncio.run(mode.run(run=run, user_input={}))

    def test_rerun_with_identical_input_skips_stage_invocation(self, run) -> None:
        """A second identical ``Mode.run`` hits the ledger and re-invokes no
        stage bodies (verified content-addressed cache)."""
        counter: dict[str, int] = {}
        stages = [
            CountingStage("StageA", counter, kind="user_plan"),
            CountingStage("StageB", counter, kind="experiment_report"),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "cache-me"}

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 1}

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 1}, (
            "identical Mode.run re-execution must hit the cache and skip stage bodies"
        )

    def test_resume_does_not_reinvoke_completed_stages(self, run) -> None:
        """After a mid-pipeline failure, the re-run resumes from the failed stage:
        completed upstream stages are not re-invoked; the failed stage and its
        downstream are."""
        counter: dict[str, int] = {}
        stages = [
            CountingStage("StageA", counter, kind="user_plan"),
            FailOnceStage("StageB", counter),
            CountingStage("StageC", counter, kind="final_report"),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "resume-me"}

        with pytest.raises(StageExecutionError):
            asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter["StageA"] == 1
        assert counter["StageB"] == 1
        assert counter.get("StageC", 0) == 0  # never reached past the failure

        result = asyncio.run(mode.run(run=run, user_input=user_input))

        assert counter["StageA"] == 1, "completed upstream stage must not re-run on resume"
        assert counter["StageB"] == 2, "failed stage must be re-attempted on resume"
        assert counter["StageC"] == 1, "downstream stage runs after the resumed stage succeeds"
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "final_report"

    def test_ledger_is_self_describing(self, run) -> None:
        """The completion ledger names the Run, the mode, and each stage →
        artifact id + producing-code fingerprint (the verified-resume key) — the
        Run-side of the provenance linkage into the harness artifact world."""
        counter: dict[str, int] = {}
        mode = _make_mode(
            lambda _ui: [
                CountingStage("StageA", counter, kind="user_plan"),
                CountingStage("StageB", counter, kind="final_report"),
            ]
        )
        result = asyncio.run(mode.run(run=run, user_input={"goal": "link-me"}))

        ledger = json.loads(_ledger_path(run).read_text(encoding="utf-8"))
        assert ledger["run_id"] == run.id
        assert ledger["mode"] == "demo"
        assert set(ledger["stages"]) == {"StageA", "StageB"}
        assert ledger["stages"]["StageA"]["artifact"] == result.stage_artifacts[0].id
        assert ledger["stages"]["StageB"]["artifact"] == result.stage_artifacts[1].id
        for entry in ledger["stages"].values():
            assert entry["fingerprint"].startswith("sha256:")

    def test_recomputes_unverifiable_legacy_ledger_entries(self, run) -> None:
        """Pre-fingerprint ledger entries (bare artifact-id strings) cannot prove
        the producing code is current → dropped with a warning and re-run once;
        the rewritten fingerprinted ledger then skips again."""
        counter: dict[str, int] = {}
        stages = [
            CountingStage("StageA", counter, kind="user_plan"),
            CountingStage("StageB", counter, kind="final_report"),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "legacy-ledger"}

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 1}

        ledger_path = _ledger_path(run)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        flat = {name: entry["artifact"] for name, entry in ledger["stages"].items()}
        ledger_path.write_text(json.dumps(flat), encoding="utf-8")

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 2, "StageB": 2}, (
            "unverifiable legacy entries must recompute exactly once"
        )

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 2, "StageB": 2}

    def test_recomputes_stage_whose_fingerprint_changed(self, run) -> None:
        """A ledger entry whose stage-code fingerprint mismatches recomputes;
        entries that still verify keep skipping."""
        counter: dict[str, int] = {}
        stages = [
            CountingStage("StageA", counter, kind="user_plan"),
            CountingStage("StageB", counter, kind="final_report"),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "fingerprint-me"}

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 1}

        ledger_path = _ledger_path(run)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["stages"]["StageB"]["fingerprint"] = "sha256:stale"
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 2}, (
            "the code-changed stage must recompute; the verified stage must not"
        )

    def test_recomputes_stage_whose_artifact_is_gone(self, run) -> None:
        """A ledger entry pointing at a missing artifact recomputes its stage;
        the intact stage does not."""
        counter: dict[str, int] = {}
        stages = [
            CountingStage("StageA", counter, kind="user_plan"),
            CountingStage("StageB", counter, kind="final_report"),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "lost-artifact"}

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 1, "StageB": 1}

        ledger_path = _ledger_path(run)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["stages"]["StageA"]["artifact"] = "art-nonexistent"
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

        asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"StageA": 2, "StageB": 1}, (
            "the artifact-less stage must recompute; the intact stage must not"
        )
