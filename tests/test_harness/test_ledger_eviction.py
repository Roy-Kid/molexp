"""Ledger eviction on validation rejection — ``Mode``'s anti-deadlock contract.

Production failure this guards against: an LLM generator stage succeeds and is
recorded in the Mode completion ledger, then its *validator* stage rejects the
generated artifact (``StagePersistedFailureError``) and the run fails. Without
eviction every subsequent ``Mode.run`` resume skips the generator (ledger hit
pins the rejected artifact) and the deterministic validator re-fails in
seconds — a permanent deadlock with no repair path.

The contract (``Mode._evict_rejected_producers``): on a *persisted* failure,
the ledger entries for the nearest recorded producers of the rejected
artifact(s) — and every entry at a later pipeline position — are evicted
persistently, in the ledger file, on the failure path, so the next run
regenerates instead of re-failing forever. Generic (non-persisted) failures
evict nothing: the failed stage was never recorded and simply re-runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import PlanArtifactRef
from molexp.workspace import Workspace

# ───────────────────────────────────────────────────────── fixtures / stubs


@pytest.fixture()
def run(tmp_path: Path):
    """A fresh, materialized ``workspace.Run`` under a tmp workspace."""
    ws = Workspace(tmp_path / "lab", name="eviction-lab")
    ws.materialize()
    project = ws.add_project("demo")
    exp = project.add_experiment("train")
    return exp.add_run(params={"seed": 0})


def _make_mode(stage_factory):
    """Build a concrete Mode whose ``stages`` returns ``stage_factory(user_input)``."""
    from molexp.harness.mode import Mode

    class _DemoMode(Mode):
        name = "demo"

        def stages(self, user_input):
            return stage_factory(user_input)

    return _DemoMode()


def _read_ledger(run) -> dict:
    ledger_files = list((Path(run.run_dir) / ".mode_ledger").glob("*.json"))
    assert len(ledger_files) == 1
    return json.loads(ledger_files[0].read_text(encoding="utf-8"))


class GeneratorStage(Stage):
    """Counting generator: each call produces a distinct artifact of ``kind``."""

    name: ClassVar[str] = "generator"

    def __init__(self, name: str, counter: dict[str, int], kind: str = "thing") -> None:
        self.name = name
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


class RejectStaleValidator(Stage):
    """Persist-then-raise validator: rejects ``kind`` artifacts with ``calls < ok_from``.

    Mirrors the production validators (ValidateTestSpec / ValidateTestSource):
    the persisted report's ``parent_ids`` name the rejected artifact. Acceptance
    keys on the *artifact content*, so it models "the stale artifact is
    rejected; a regenerated one is valid".
    """

    name: ClassVar[str] = "validator"

    def __init__(self, name: str, kind: str = "thing", ok_from: int = 2) -> None:
        self.name = name
        self._kind = kind
        self._ok_from = ok_from

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        target = ctx.artifact_store.latest_by_kind(self._kind)
        assert target is not None
        payload = json.loads(ctx.artifact_store.get(target.id))
        if payload["calls"] < self._ok_from:
            report = ctx.artifact_store.put_json(
                kind="validation_report",
                obj={"passed": False, "violations": ["unknown_task_target"]},
                created_by=self.name,
                parent_ids=[target.id],
            )
            raise StagePersistedFailureError(report, "validation failed: unknown_task_target")
        return ctx.artifact_store.put_json(
            kind="validation_report",
            obj={"passed": True, "violations": []},
            created_by=self.name,
            parent_ids=[target.id],
        )


class TestModeLedgerEviction:
    """``Mode`` evicts ledgered producers of a validator-rejected artifact."""

    def test_rerun_after_rejection_regenerates_instead_of_deadlocking(self, run) -> None:
        """The production deadlock, distilled: generator succeeds + is ledgered,
        validator rejects, run fails. The NEXT run must re-invoke the generator
        (regenerating the artifact) and succeed — not skip it and re-fail forever."""
        counter: dict[str, int] = {}
        stages = [
            GeneratorStage("gen", counter, kind="thing"),
            RejectStaleValidator("val", kind="thing", ok_from=2),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "no-deadlock"}

        with pytest.raises(StagePersistedFailureError):
            asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter["gen"] == 1

        # Without eviction this second run skips "gen" (ledger hit) and re-fails
        # on the same pinned artifact — the observed permanent deadlock.
        result = asyncio.run(mode.run(run=run, user_input=user_input))

        assert counter["gen"] == 2, "resume must regenerate the rejected artifact"
        assert result.final_artifact is not None

    def test_eviction_drops_producer_and_suffix_but_keeps_prefix(self, run) -> None:
        """The eviction window: entries *before* the rejected producer survive,
        while the producer and every entry at a later pipeline position are
        evicted (a linear pipeline's downstream may have consumed the evicted
        output) and re-run on resume."""
        counter: dict[str, int] = {}
        stages = [
            GeneratorStage("keep", counter, kind="log"),
            GeneratorStage("alpha", counter, kind="alpha"),
            GeneratorStage("beta", counter, kind="beta"),
            RejectStaleValidator("val", kind="alpha", ok_from=2),
        ]
        mode = _make_mode(lambda _ui: stages)
        user_input = {"goal": "window"}

        with pytest.raises(StagePersistedFailureError):
            asyncio.run(mode.run(run=run, user_input=user_input))
        ledger = _read_ledger(run)
        assert "keep" in ledger["stages"], "entries before the rejected producer must survive"
        assert "alpha" not in ledger["stages"], "the rejected artifact's producer must be evicted"
        assert "beta" not in ledger["stages"], "downstream entries must be evicted too"

        result = asyncio.run(mode.run(run=run, user_input=user_input))
        assert counter == {"keep": 1, "alpha": 2, "beta": 2}, (
            "the prefix must not re-run; producer + suffix must regenerate on resume"
        )
        assert result.final_artifact is not None

    def test_eviction_walks_past_unledgered_intermediates(self, run) -> None:
        """A rejected artifact may itself be unledgered (e.g. a repair-loop
        attempt); eviction walks its ancestry to the nearest ledgered producer."""
        counter: dict[str, int] = {}

        class RejectViaIntermediate(Stage):
            name: ClassVar[str] = "loop"

            async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
                thing = ctx.artifact_store.latest_by_kind("thing")
                assert thing is not None
                attempt = ctx.artifact_store.put_json(
                    kind="attempt",
                    obj={"derived": True},
                    created_by="loop",
                    parent_ids=[thing.id],
                )
                report = ctx.artifact_store.put_json(
                    kind="validation_report",
                    obj={"passed": False, "violations": ["unknown_task_target"]},
                    created_by="loop",
                    parent_ids=[attempt.id],
                )
                raise StagePersistedFailureError(report, "exhausted repair attempts")

        stages = [GeneratorStage("gen", counter, kind="thing"), RejectViaIntermediate()]
        mode = _make_mode(lambda _ui: stages)

        with pytest.raises(StagePersistedFailureError):
            asyncio.run(mode.run(run=run, user_input={"goal": "walk"}))

        ledger = _read_ledger(run)
        assert "gen" not in ledger["stages"], (
            "eviction must walk through unledgered intermediate artifacts to the "
            "nearest ledger-recorded producer"
        )

    def test_generic_failure_evicts_nothing(self, run) -> None:
        """A non-persisted failure identifies no rejected artifact: completed
        entries stay, preserving resume-from-the-failed-stage semantics."""
        counter: dict[str, int] = {}

        class Boom(Stage):
            name: ClassVar[str] = "boom"

            async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
                raise RuntimeError("transient")

        stages = [GeneratorStage("gen", counter, kind="thing"), Boom()]
        mode = _make_mode(lambda _ui: stages)

        with pytest.raises(StageExecutionError):
            asyncio.run(mode.run(run=run, user_input={"goal": "generic"}))

        ledger = _read_ledger(run)
        assert "gen" in ledger["stages"], "generic failures must not evict completed entries"
