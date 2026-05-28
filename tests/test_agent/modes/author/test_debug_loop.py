"""Tests for the per-task subprocess debug loop.

The loop runs each generated task's pytest test in a genuine isolated
subprocess via :class:`~molexp.agent.execution_env.LocalExecutionEnv`.
On failure it asks the LLM for a
:class:`~molexp.agent.modes.author.codegen.RepairDecision` — diagnose
the root cause then patch the impl, the test, or both. Each patched
file is gated (impl: shape + evidence; test: syntax + evidence) before
landing on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.modes.author.codegen import RepairDecision, TaskImplDraft
from molexp.agent.modes.author.debug_loop import (
    run_subprocess_test,
    run_task_debug_loop,
)
from molexp.agent.router import RouterTextResult
from molexp.agent.types import UsageBreakdown

from .conftest import make_plan_graph


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _impl_decision(
    *,
    imports: tuple[str, ...] = (),
    body: str = "pass",
    diagnosis: str = "impl bug",
) -> RepairDecision:
    """Build a RepairDecision that patches only the impl."""
    return RepairDecision(
        diagnosis=diagnosis,
        impl=TaskImplDraft(imports=imports, body=body),
    )


def _test_decision(
    *,
    test_source: str,
    diagnosis: str = "test bug",
) -> RepairDecision:
    """Build a RepairDecision that patches only the test."""
    return RepairDecision(diagnosis=diagnosis, test_source=test_source)


# ── run_subprocess_test — the isolated runner ────────────────────────────


def test_passing_test_exits_in_one_attempt(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    work = tmp_path / "work"
    test = _write(work / "test_ok.py", "def test_ok() -> None:\n    assert 1 == 1\n")
    outcome = run_subprocess_test(execution_env=env, test_path=test, cwd=work, timeout=30.0)
    assert outcome.passed
    assert outcome.exit_code == 0


def test_failing_test_reports_failure(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    work = tmp_path / "work"
    test = _write(work / "test_bad.py", "def test_bad() -> None:\n    assert 1 == 2\n")
    outcome = run_subprocess_test(execution_env=env, test_path=test, cwd=work, timeout=30.0)
    assert not outcome.passed
    assert not outcome.timed_out


def test_sleeping_test_is_killed_by_timeout(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    work = tmp_path / "work"
    test = _write(
        work / "test_slow.py",
        "import time\n\n\ndef test_slow() -> None:\n    time.sleep(30)\n",
    )
    outcome = run_subprocess_test(execution_env=env, test_path=test, cwd=work, timeout=1.0)
    assert not outcome.passed
    assert outcome.timed_out


def test_subprocess_runs_with_confined_cwd(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    work = tmp_path / "confined"
    test = _write(
        work / "test_cwd.py",
        "import os\nfrom pathlib import Path\n\n\n"
        "def test_cwd() -> None:\n"
        f"    assert Path(os.getcwd()).resolve() == Path({str(work)!r}).resolve()\n",
    )
    outcome = run_subprocess_test(execution_env=env, test_path=test, cwd=work, timeout=30.0)
    assert outcome.passed


# ── run_task_debug_loop — the run→repair→re-run cycle ────────────────────


class _NoRepairRouter:
    """A router whose repair call is never expected to be used."""

    async def complete_text(self, **kw: object) -> RouterTextResult:
        return RouterTextResult(text="")

    async def complete_structured(self, **kw: object) -> RepairDecision:
        raise AssertionError("debug loop asked for a repair when none was expected")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _ScriptedDraftRouter:
    """A router that returns a sequence of RepairDecision fixes."""

    def __init__(self, decisions: list[RepairDecision]) -> None:
        self._decisions = list(decisions)
        self.repair_calls = 0

    async def complete_text(self, **kw: object) -> RouterTextResult:
        return RouterTextResult(text="")

    async def complete_structured(self, **kw: object) -> RepairDecision:
        self.repair_calls += 1
        if self._decisions:
            return self._decisions.pop(0)
        return _impl_decision(body="result = None")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


@pytest.mark.asyncio
async def test_debug_loop_converges_in_one_attempt_when_test_passes(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    src = tmp_path / "src"
    impl = _write(src / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(src / "tests" / "test_run.py", "def test_run() -> None:\n    assert True\n")
    result = await run_task_debug_loop(
        task_id="run",
        impl_path=impl,
        test_path=test,
        plan_graph=make_plan_graph(),
        router=_NoRepairRouter(),  # type: ignore[arg-type]
        execution_env=env,
        src_root=src,
        debug_attempts=3,
    )
    assert result.converged
    assert result.attempts == 1
    assert result.diffs == ()


@pytest.mark.asyncio
async def test_debug_loop_exhausts_budget_and_plants_repair_signal(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    src = tmp_path / "src"
    impl = _write(src / "experiment" / "tasks" / "run.py", "VALUE = 1\n")
    # The test always asserts False — every repair is rejected by the
    # test, regardless of body content.
    test = _write(
        src / "tests" / "test_run.py",
        "def test_run() -> None:\n    assert False\n",
    )
    router = _ScriptedDraftRouter(
        [
            _impl_decision(body="result = None"),
            _impl_decision(body="result = None"),
        ]
    )
    result = await run_task_debug_loop(
        task_id="run",
        impl_path=impl,
        test_path=test,
        plan_graph=make_plan_graph(),
        router=router,  # type: ignore[arg-type]
        execution_env=env,
        src_root=src,
        debug_attempts=3,
    )
    assert not result.converged
    assert result.attempts == 3
    # Every iteration plus a final budget-exhaustion diff are recorded.
    assert len(result.diffs) >= 1
    for diff in result.diffs:
        assert diff.failed_invariant
        assert diff.affected_nodes == ("run",)


@pytest.mark.asyncio
async def test_debug_loop_converges_on_second_attempt(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    src = tmp_path / "src"
    impl = _write(src / "experiment" / "tasks" / "run.py", "VALUE = 1\n")
    # The test imports the assembled ``run`` function and asserts its
    # returned dict carries result=2.
    test = _write(
        src / "tests" / "test_run.py",
        "import asyncio\n"
        "from types import SimpleNamespace\n"
        "from experiment.tasks.run import run\n\n\n"
        "def test_run() -> None:\n"
        "    ctx = SimpleNamespace(inputs={'payload': None})\n"
        "    out = asyncio.run(run(ctx))\n"
        "    assert out['result'] == 2\n",
    )
    router = _ScriptedDraftRouter([_impl_decision(body="result = 2")])
    result = await run_task_debug_loop(
        task_id="run",
        impl_path=impl,
        test_path=test,
        plan_graph=make_plan_graph(),
        router=router,  # type: ignore[arg-type]
        execution_env=env,
        src_root=src,
        debug_attempts=3,
    )
    assert result.converged
    assert result.attempts == 2
    assert router.repair_calls == 1
    assert len(result.diffs) == 1


# ── _apply_targeted_fix repair-callable routing ───────────────────────────


@pytest.mark.asyncio
async def test_apply_targeted_fix_uses_grounded_repair_when_provided(
    tmp_path: Path,
) -> None:
    """When a `repair` callable is supplied, it replaces the no-tool router
    path entirely — the source-grounded repair takes over."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(
        tmp_path / "src" / "tests" / "test_run.py",
        "def test() -> None:\n    assert True\n",
    )

    repair_calls: list[str] = []

    async def grounded_repair(prompt: str) -> RepairDecision:
        repair_calls.append(prompt)
        return _impl_decision(body="result = 42", diagnosis="impl missed the return value")

    class _ExplodingRouter:
        """If the fallback router path is hit, the test fails loudly."""

        async def complete_structured(self, **_kw: object) -> RepairDecision:
            raise AssertionError("router fallback used despite repair= being supplied")

        async def complete_text(self, **_kw: object) -> RouterTextResult:  # pragma: no cover
            raise AssertionError("complete_text not expected")

        def clear_usage(self) -> None:
            return None

        def snapshot_usage(self) -> UsageBreakdown:
            return UsageBreakdown()

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="AttributeError: module 'molpy' has no attribute 'forcefield'",
        plan_graph=make_plan_graph(),
        router=_ExplodingRouter(),  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=grounded_repair,
    )

    assert repair_calls, "grounded repair callable should have been invoked"
    assert "AttributeError" in repair_calls[0]
    written = impl.read_text()
    assert "async def run(ctx):" in written
    assert "result = 42" in written
    assert "return {'result': result}" in written


@pytest.mark.asyncio
async def test_apply_targeted_fix_falls_back_to_router_when_repair_is_none(
    tmp_path: Path,
) -> None:
    """`repair=None` keeps the no-tool router.complete_structured path."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(
        tmp_path / "src" / "tests" / "test_run.py",
        "def test() -> None:\n    assert True\n",
    )

    router = _ScriptedDraftRouter(
        [_impl_decision(body="result = 7", diagnosis="impl missed the return value")]
    )

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="boom",
        plan_graph=make_plan_graph(),
        router=router,  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=None,
    )

    assert router.repair_calls == 1
    written = impl.read_text()
    assert "async def run(ctx):" in written
    assert "result = 7" in written


# ── gate exhaustion ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_targeted_fix_raises_when_gate_keeps_rejecting(
    tmp_path: Path,
) -> None:
    """A repair that keeps emitting test code is rejected every retry; on
    exhaustion the function raises ``_RepairGateExhausted`` rather than
    writing the broken source to disk."""
    from molexp.agent.modes.author.debug_loop import (
        _apply_targeted_fix,
        _RepairGateExhausted,
    )
    from molexp.agent.router import ModelTier

    impl_text = "async def run(ctx):\n    result = 1\n    return {'result': result}\n"
    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", impl_text)
    test = _write(
        tmp_path / "src" / "tests" / "test_run.py",
        "def test() -> None:\n    assert True\n",
    )

    repair_calls = 0

    async def bad_repair(prompt: str) -> RepairDecision:
        nonlocal repair_calls
        repair_calls += 1
        # The draft imports a symbol from the same tracked namespace as
        # the plan's api_refs (``molpy``) but which is NOT on the allow-
        # list — the codegen-evidence gate rejects it every attempt.
        return RepairDecision(
            diagnosis="impl uses Fake",
            impl=TaskImplDraft(
                imports=("from molpy.bogus_module import Fake",),
                body="result = Fake()",
            ),
        )

    with pytest.raises(_RepairGateExhausted):
        await _apply_targeted_fix(
            task_id="run",
            impl_path=impl,
            test_path=test,
            traceback="boom",
            plan_graph=make_plan_graph(),
            router=_NoRepairRouter(),  # type: ignore[arg-type]
            tier=ModelTier.DEFAULT,
            repair=bad_repair,
        )

    # The on-disk impl is preserved — broken repair never landed.
    assert impl.read_text() == impl_text
    # The inline retry exercised the gate-retry budget.
    assert repair_calls >= 2


@pytest.mark.asyncio
async def test_apply_targeted_fix_feeds_gate_verdict_to_next_prompt(
    tmp_path: Path,
) -> None:
    """The second repair-prompt contains the gate verdict from the first
    attempt so the model can fix the specific issue rather than guess."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(tmp_path / "src" / "tests" / "test_run.py", "def test(): assert True\n")

    prompts_seen: list[str] = []

    async def two_step_repair(prompt: str) -> RepairDecision:
        prompts_seen.append(prompt)
        if len(prompts_seen) == 1:
            # First attempt — un-evidenced symbol fails the evidence gate.
            return RepairDecision(
                diagnosis="trying Fake",
                impl=TaskImplDraft(
                    imports=("from molpy.bogus_module import Fake",),
                    body="result = Fake()",
                ),
            )
        # Second attempt — accept the feedback and emit a valid draft.
        return _impl_decision(body="result = 9", diagnosis="impl now binds result correctly")

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="initial test failure",
        plan_graph=make_plan_graph(),
        router=_NoRepairRouter(),  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=two_step_repair,
    )

    assert len(prompts_seen) == 2
    assert "REJECTED BY CODEGEN GATE" in prompts_seen[1]
    written = impl.read_text()
    assert "async def run(ctx):" in written
    assert "result = 9" in written


# ── allowlist + timeout + generic-exception regressions ──────────────────


@pytest.mark.asyncio
async def test_apply_targeted_fix_prompt_carries_allowlist(tmp_path: Path) -> None:
    """Repair prompt must include the ``ALLOWED PROJECT IMPORTS`` block
    so the model has positive guidance when the evidence gate rejects.
    Without it, retries were blind swaps that exhausted the budget."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(tmp_path / "src" / "tests" / "test_run.py", "def test(): assert True\n")

    captured: list[str] = []

    async def echo_repair(prompt: str) -> RepairDecision:
        captured.append(prompt)
        return _impl_decision(body="result = 1", diagnosis="impl now binds result")

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="boom",
        plan_graph=make_plan_graph(),
        router=_NoRepairRouter(),  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=echo_repair,
    )

    assert captured, "repair callable must be invoked"
    prompt = captured[0]
    assert "ALLOWED PROJECT IMPORTS" in prompt
    # make_plan_graph() seeds api_refs=("molpy.System",) on every step.
    assert "molpy.System" in prompt


@pytest.mark.asyncio
async def test_apply_targeted_fix_fallback_router_respects_wall_clock(
    tmp_path: Path,
) -> None:
    """``repair=None`` fallback path is also wrapped in ``asyncio.wait_for``
    so a hung provider can't wedge the debug loop indefinitely."""
    from molexp.agent.modes.author import debug_loop
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(tmp_path / "src" / "tests" / "test_run.py", "def test(): assert True\n")

    class _HangingRouter:
        """``complete_structured`` blocks for longer than the timeout."""

        async def complete_text(self, **_kw: object) -> RouterTextResult:  # pragma: no cover
            raise AssertionError("complete_text not expected")

        async def complete_structured(self, **_kw: object) -> RepairDecision:
            import asyncio as _aio

            await _aio.sleep(5.0)
            raise AssertionError("router ran past the wall-clock budget")

        def clear_usage(self) -> None:
            return None

        def snapshot_usage(self) -> UsageBreakdown:
            return UsageBreakdown()

    # Shrink the fallback budget for the test — same mechanism, tiny window.
    saved = debug_loop._FALLBACK_REPAIR_TIMEOUT_SECONDS
    debug_loop._FALLBACK_REPAIR_TIMEOUT_SECONDS = 0.05
    try:
        with pytest.raises(TimeoutError):
            await _apply_targeted_fix(
                task_id="run",
                impl_path=impl,
                test_path=test,
                traceback="hang",
                plan_graph=make_plan_graph(),
                router=_HangingRouter(),  # type: ignore[arg-type]
                tier=ModelTier.DEFAULT,
                repair=None,
            )
    finally:
        debug_loop._FALLBACK_REPAIR_TIMEOUT_SECONDS = saved


@pytest.mark.asyncio
async def test_debug_loop_isolates_non_timeout_exception_to_one_task(
    tmp_path: Path,
) -> None:
    """A non-timeout, non-gate exception from the repair callable must be
    recorded as one task's failed iteration (not propagated), so a
    sibling task running in parallel via ``asyncio.gather`` isn't
    cancelled by an unrelated error in another task's repair."""
    from molexp.agent.execution_env import LocalExecutionEnv

    env = LocalExecutionEnv(scratch_dir=tmp_path / "scratch")
    src = tmp_path / "src"
    impl = _write(src / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(src / "tests" / "test_run.py", "def test(): assert False\n")

    async def exploding_repair(prompt: str) -> RepairDecision:
        raise RuntimeError("simulated pydantic-ai UnexpectedModelBehavior")

    result = await run_task_debug_loop(
        task_id="run",
        impl_path=impl,
        test_path=test,
        plan_graph=make_plan_graph(),
        router=_NoRepairRouter(),  # type: ignore[arg-type]
        execution_env=env,
        src_root=src,
        debug_attempts=3,
        repair=exploding_repair,
    )

    assert not result.converged
    assert "RuntimeError" in result.final_outcome.traceback
    # A repair_proposed diff is recorded so the caller can surface it.
    assert result.diffs


# ── RepairDecision: test-source patching ─────────────────────────────────


@pytest.mark.asyncio
async def test_apply_targeted_fix_can_rewrite_the_test(tmp_path: Path) -> None:
    """When the repair decides the TEST is the bug (e.g. broken stub
    class), it emits ``test_source`` and the loop rewrites the test
    file. The impl is preserved untouched."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl_text = "async def run(ctx):\n    return {'result': 1}\n"
    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", impl_text)
    test = _write(
        tmp_path / "src" / "tests" / "test_run.py",
        "class StubAtomType:\n"
        "    __slots__ = ()\n"
        "    def __init__(self, idx):\n"
        "        self.idx = idx  # bug: empty __slots__ forbids attributes\n",
    )

    fixed_test = (
        "import asyncio\n"
        "from types import SimpleNamespace\n"
        "from run import run\n\n"
        "def test_run():\n"
        "    out = asyncio.run(run(SimpleNamespace(inputs=None)))\n"
        "    assert out['result'] == 1\n"
    )

    async def repair_test(prompt: str) -> RepairDecision:
        return RepairDecision(
            diagnosis="test's stub class has empty __slots__ then tries to set attributes",
            test_source=fixed_test,
        )

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="AttributeError: 'StubAtomType' object has no attribute 'idx'",
        plan_graph=make_plan_graph(),
        router=_NoRepairRouter(),  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=repair_test,
    )

    # Impl preserved verbatim.
    assert impl.read_text() == impl_text
    # Test rewritten.
    assert test.read_text() == fixed_test


@pytest.mark.asyncio
async def test_apply_targeted_fix_rejects_empty_decision(tmp_path: Path) -> None:
    """A RepairDecision with neither impl nor test_source is a no-op; the
    loop rejects it and feeds the verdict to the next attempt."""
    from molexp.agent.modes.author.debug_loop import (
        _apply_targeted_fix,
        _RepairGateExhausted,
    )
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(tmp_path / "src" / "tests" / "test_run.py", "def test(): assert True\n")

    prompts_seen: list[str] = []

    async def empty_repair(prompt: str) -> RepairDecision:
        prompts_seen.append(prompt)
        return RepairDecision(diagnosis="hmm")  # neither impl nor test_source

    with pytest.raises(_RepairGateExhausted):
        await _apply_targeted_fix(
            task_id="run",
            impl_path=impl,
            test_path=test,
            traceback="boom",
            plan_graph=make_plan_graph(),
            router=_NoRepairRouter(),  # type: ignore[arg-type]
            tier=ModelTier.DEFAULT,
            repair=empty_repair,
        )

    # The retry budget was exhausted by empty decisions.
    assert len(prompts_seen) >= 2
    # The second prompt carries feedback that the previous decision was empty.
    assert "RepairDecision is empty" in prompts_seen[1]


@pytest.mark.asyncio
async def test_apply_targeted_fix_requires_diagnosis(tmp_path: Path) -> None:
    """A RepairDecision with empty diagnosis is rejected — the model
    must name the root cause before patching."""
    from molexp.agent.modes.author.debug_loop import (
        _apply_targeted_fix,
        _RepairGateExhausted,
    )
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(tmp_path / "src" / "tests" / "test_run.py", "def test(): assert True\n")

    async def no_diagnosis(prompt: str) -> RepairDecision:
        return RepairDecision(
            diagnosis="   ",  # whitespace-only
            impl=TaskImplDraft(body="result = 1"),
        )

    with pytest.raises(_RepairGateExhausted):
        await _apply_targeted_fix(
            task_id="run",
            impl_path=impl,
            test_path=test,
            traceback="boom",
            plan_graph=make_plan_graph(),
            router=_NoRepairRouter(),  # type: ignore[arg-type]
            tier=ModelTier.DEFAULT,
            repair=no_diagnosis,
        )
