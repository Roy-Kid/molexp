"""``Mode`` orchestration tests (spec plan-mode-revival-01).

``Mode`` is the thin harness base class that compiles a declared list of
``Stage`` objects into a ``molexp.workflow`` ``Workflow`` (one ``StageTask``
per stage, linearly chained) and executes it on a ``workspace.Run``,
returning a frozen ``ModeResult``.

    class Mode(ABC):
        name: ClassVar[str]
        def stages(self, user_input) -> list[Stage]: ...
        async def run(self, *, run, user_input, gateway=None) -> ModeResult: ...

Tests map to acceptance:

- ac-007  happy path — declared stages execute on a tmp ``workspace.Run``;
          ``ModeResult`` carries each stage's ``ArtifactRef`` + the final.
- ac-009  cache-hit — a second identical ``Mode.run`` leaves an
          invocation-counting stub Stage's counter flat.
- ac-010  resume — a run failing at stage N, re-run, does NOT re-invoke
          stages ``0..N-1``.

ENGINE-WIRING ASSUMPTION (load-bearing for ac-009 / ac-010): the
``molexp.workflow`` runtime does NOT, today, engage content-addressed
caching (``Caching`` is a standalone utility) or checkpoint-resume
(``WorkflowRuntime`` docstring states ``resume()`` is removed and
per-frame snapshots are no longer injected into the graph runner). The only
engine-native skip mechanism is ``Workflow.execute(seed_outputs=...)``.
Therefore the cache/resume *observable behaviour* asserted here (counter
stays flat on identical re-run; stages ``0..N-1`` not re-invoked after a
mid-pipeline failure) MUST be wired by the GREEN implementation inside
``Mode.run`` / ``StageTask`` (e.g. consulting ``ws.cache.as_cache_store()``
keyed on the stage snapshot + user_input, and/or seeding already-completed
stages via ``seed_outputs`` after reading prior execution artifacts). These
tests assert the intended behaviour; the GREEN step confirms the chosen
wiring delivers it. All tests are RED now (``Mode`` does not yet exist).
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
    return exp.add_run(parameters={"seed": 0})


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
