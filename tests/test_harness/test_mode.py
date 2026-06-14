"""``Mode`` orchestration tests.

``Mode`` is the harness base class that runs a declared list of ``Stage``
objects eagerly — one at a time through the shared audit bracket — against a
``workspace.Run``, returning a frozen ``ModeResult``.

    class Mode(ABC):
        name: ClassVar[str]
        def stages(self, user_input) -> list[Stage]: ...
        async def run(self, *, run, user_input, gateway=None) -> ModeResult: ...

Covered behaviour:

- happy path — declared stages execute; ``ModeResult`` carries each stage's
  ``ArtifactRef`` + the final.
- cache-hit — a second identical ``Mode.run`` leaves an invocation-counting
  stub Stage's counter flat.
- resume — a run failing at stage N, re-run, does NOT re-invoke stages
  ``0..N-1``.
- verified ledger — entries are reused only when the recorded stage code
  fingerprint matches and the artifact still exists; unverifiable entries
  (legacy ledgers, tampered fingerprints, missing artifacts) recompute with
  a warning — never silently reuse, never error.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import ArtifactRef
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
    """Stub stage that bumps a shared counter on every invocation.

    Cross-process determinism is not needed; the counter is an in-process
    dict so cache-hit / resume assertions can read it directly.
    """

    def __init__(self, name: str, counter: dict[str, int], kind: str = "log") -> None:
        self.name = name  # instance-level override of the ClassVar
        self._counter = counter
        self._kind = kind

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
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

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
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


# ───────────────────────────────────────────── ac-007: happy path → ModeResult


def test_mode_run_executes_all_stages_and_returns_mode_result(run) -> None:
    """ac-007: declared stages run; ModeResult carries each ref + the final."""
    from molexp.harness.schemas import ModeResult

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
    # Every stage was invoked exactly once.
    assert counter == {"StageA": 1, "StageB": 1, "StageC": 1}
    # One ArtifactRef per stage, in declared order.
    assert len(result.stage_artifacts) == 3
    assert [a.kind for a in result.stage_artifacts] == [
        "user_plan",
        "experiment_report",
        "final_report",
    ]
    # The final artifact is the last stage's product.
    assert result.final_artifact is not None
    assert result.final_artifact.kind == "final_report"


def test_mode_run_rejects_empty_stage_list(run) -> None:
    """ac-007 edge case: an empty stage list is rejected."""
    mode = _make_mode(lambda _ui: [])
    with pytest.raises((ValueError, StageExecutionError)):
        asyncio.run(mode.run(run=run, user_input={}))


def test_mode_run_single_stage_pipeline(run) -> None:
    """ac-007 edge case: a single-stage pipeline produces one artifact + final."""
    counter: dict[str, int] = {}
    mode = _make_mode(lambda _ui: [CountingStage("Solo", counter, kind="log")])

    result = asyncio.run(mode.run(run=run, user_input={}))

    assert counter == {"Solo": 1}
    assert len(result.stage_artifacts) == 1
    assert result.final_artifact is not None
    assert result.final_artifact.id == result.stage_artifacts[0].id


# ──────────────────────────────────────── ac-009: identical re-run hits cache


def test_mode_rerun_with_identical_input_skips_stage_invocation(run) -> None:
    """ac-009: a 2nd identical Mode.run leaves the invocation counter flat.

    See module docstring — this asserts the intended cache-hit observable;
    the GREEN wiring (workspace-backed content-addressed cache and/or
    seed_outputs) must deliver it.
    """
    counter: dict[str, int] = {}
    stages = [
        CountingStage("StageA", counter, kind="user_plan"),
        CountingStage("StageB", counter, kind="experiment_report"),
    ]
    mode = _make_mode(lambda _ui: stages)

    user_input = {"goal": "cache-me"}
    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 1}

    # Re-running identical input must NOT re-invoke the stage bodies.
    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 1}, (
        "identical Mode.run re-execution must hit the cache and skip stage bodies"
    )


# ───────────────────────────────────── ac-010: resume from the failed stage


def test_mode_resume_does_not_reinvoke_completed_stages(run) -> None:
    """ac-010: after a mid-pipeline failure, re-run resumes from the failed stage.

    Pipeline: StageA (ok) → StageB (fails first attempt) → StageC. First run
    fails inside StageB. The re-run must NOT re-invoke StageA (already
    completed before the failure) and must re-attempt StageB onward. See the
    module docstring for the engine-wiring assumption this depends on.
    """
    counter: dict[str, int] = {}
    stages = [
        CountingStage("StageA", counter, kind="user_plan"),
        FailOnceStage("StageB", counter),
        CountingStage("StageC", counter, kind="final_report"),
    ]
    mode = _make_mode(lambda _ui: stages)

    user_input = {"goal": "resume-me"}

    # First run fails mid-pipeline at StageB.
    with pytest.raises(StageExecutionError):
        asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter["StageA"] == 1
    assert counter["StageB"] == 1
    assert counter.get("StageC", 0) == 0  # never reached past the failure

    # Re-run: StageA must NOT be re-invoked (resume from the failed stage);
    # StageB succeeds on its 2nd attempt; StageC then runs.
    result = asyncio.run(mode.run(run=run, user_input=user_input))

    assert counter["StageA"] == 1, "completed upstream stage must not re-run on resume"
    assert counter["StageB"] == 2, "failed stage must be re-attempted on resume"
    assert counter["StageC"] == 1, "downstream stage runs after the resumed stage succeeds"
    assert result.final_artifact is not None
    assert result.final_artifact.kind == "final_report"


# ─────────────────────── ledger as the Run ↔ harness-artifact linkage record


def test_mode_ledger_is_self_describing(run) -> None:
    """The completion ledger names the Run, the mode, and stage → artifact ids.

    The ledger lives under ``run_dir/.mode_ledger`` — it is the workspace-Run
    side of the provenance linkage: from a Run on disk you can discover which
    pipeline ran and which artifact each stage produced.
    """
    import json

    counter: dict[str, int] = {}
    mode = _make_mode(
        lambda _ui: [
            CountingStage("StageA", counter, kind="user_plan"),
            CountingStage("StageB", counter, kind="final_report"),
        ]
    )
    user_input = {"goal": "link-me"}
    result = asyncio.run(mode.run(run=run, user_input=user_input))

    ledger_files = list((Path(run.run_dir) / ".mode_ledger").glob("*.json"))
    assert len(ledger_files) == 1
    ledger = json.loads(ledger_files[0].read_text(encoding="utf-8"))
    assert ledger["run_id"] == run.id
    assert ledger["mode"] == "demo"
    assert set(ledger["stages"]) == {"StageA", "StageB"}
    # Each entry pairs the produced artifact with the producing stage's
    # code fingerprint — the verified-resume key.
    assert ledger["stages"]["StageA"]["artifact"] == result.stage_artifacts[0].id
    assert ledger["stages"]["StageB"]["artifact"] == result.stage_artifacts[1].id
    for entry in ledger["stages"].values():
        assert entry["fingerprint"].startswith("sha256:")


def test_mode_recomputes_unverifiable_legacy_ledger_entries(run) -> None:
    """Pre-fingerprint ledger entries are unverifiable → recompute, never trust.

    A legacy ledger (entries are bare artifact-id strings) cannot prove the
    stage code that produced its artifacts is the code that would run today,
    so the entries are dropped with a warning and the stages re-run once —
    after which the rewritten ledger is fingerprinted and skips again.
    """
    import json

    counter: dict[str, int] = {}
    stages = [
        CountingStage("StageA", counter, kind="user_plan"),
        CountingStage("StageB", counter, kind="final_report"),
    ]
    mode = _make_mode(lambda _ui: stages)
    user_input = {"goal": "legacy-ledger"}

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 1}

    # Rewrite the ledger in the legacy flat shape (no fingerprints), re-run.
    ledger_path = next((Path(run.run_dir) / ".mode_ledger").glob("*.json"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    flat = {name: entry["artifact"] for name, entry in ledger["stages"].items()}
    ledger_path.write_text(json.dumps(flat), encoding="utf-8")

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 2, "StageB": 2}, (
        "unverifiable legacy entries must recompute exactly once"
    )

    # The recompute re-wrote a fingerprinted ledger — a third run skips.
    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 2, "StageB": 2}


def test_mode_recomputes_stage_whose_fingerprint_changed(run) -> None:
    """A ledger entry whose stage code fingerprint mismatches recomputes.

    Only the changed stage re-runs; entries that still verify keep skipping.
    """
    import json

    counter: dict[str, int] = {}
    stages = [
        CountingStage("StageA", counter, kind="user_plan"),
        CountingStage("StageB", counter, kind="final_report"),
    ]
    mode = _make_mode(lambda _ui: stages)
    user_input = {"goal": "fingerprint-me"}

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 1}

    # Simulate StageB's code having changed since the ledger was written.
    ledger_path = next((Path(run.run_dir) / ".mode_ledger").glob("*.json"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["stages"]["StageB"]["fingerprint"] = "sha256:stale"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 2}, (
        "the code-changed stage must recompute; the verified stage must not"
    )


def test_mode_recomputes_stage_whose_artifact_is_gone(run) -> None:
    """A ledger entry pointing at a missing artifact recomputes the stage."""
    import json

    counter: dict[str, int] = {}
    stages = [
        CountingStage("StageA", counter, kind="user_plan"),
        CountingStage("StageB", counter, kind="final_report"),
    ]
    mode = _make_mode(lambda _ui: stages)
    user_input = {"goal": "lost-artifact"}

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 1, "StageB": 1}

    ledger_path = next((Path(run.run_dir) / ".mode_ledger").glob("*.json"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["stages"]["StageA"]["artifact"] = "art-nonexistent"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    asyncio.run(mode.run(run=run, user_input=user_input))
    assert counter == {"StageA": 2, "StageB": 1}, (
        "the artifact-less stage must recompute; the intact stage must not"
    )
