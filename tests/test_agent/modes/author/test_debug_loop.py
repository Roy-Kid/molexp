"""Tests for the per-task subprocess debug loop (ac-006).

The loop runs each generated task's pytest test in a genuine isolated
subprocess via :class:`~molexp.agent.harness.execution_env.LocalExecutionEnv`
— pass / always-fail / timeout / fix-on-second-attempt — with the
``cwd`` confined to the plan workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.execution_env import LocalExecutionEnv
from molexp.agent.modes.author.codegen import GeneratedModule
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
    # The test asserts os.getcwd() is the confined dir.
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

    async def complete_structured(self, **kw: object) -> GeneratedModule:
        raise AssertionError("debug loop asked for a repair when none was expected")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _ScriptedFixRouter:
    """A router that returns a sequence of fixed implementation sources."""

    def __init__(self, fixes: list[str]) -> None:
        self._fixes = list(fixes)
        self.repair_calls = 0

    async def complete_text(self, **kw: object) -> RouterTextResult:
        return RouterTextResult(text="")

    async def complete_structured(self, **kw: object) -> GeneratedModule:
        self.repair_calls += 1
        source = self._fixes.pop(0) if self._fixes else "x = 1\n"
        return GeneratedModule(task_id="run", source=source)

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
    impl = _write(src / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(
        src / "tests" / "test_run.py",
        "def test_run() -> None:\n    assert False\n",
    )
    # The repair always returns source that still doesn't fix the test.
    router = _ScriptedFixRouter(["x = 2\n", "x = 3\n"])
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
    impl = _write(
        src / "experiment" / "tasks" / "run.py",
        "VALUE = 1\n",
    )
    # The test imports VALUE and asserts it equals 2 — first impl fails.
    test = _write(
        src / "tests" / "test_run.py",
        "from experiment.tasks.run import VALUE\n\n\n"
        "def test_run() -> None:\n    assert VALUE == 2\n",
    )
    # The repair returns the corrected implementation.
    router = _ScriptedFixRouter(["VALUE = 2\n"])
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

    async def grounded_repair(prompt: str) -> GeneratedModule:
        repair_calls.append(prompt)
        return GeneratedModule(task_id="run", source="GROUNDED FIX\n")

    class _ExplodingRouter:
        """If the fallback router path is hit, the test fails loudly."""

        async def complete_structured(self, **_kw: object) -> GeneratedModule:
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
        router=_ExplodingRouter(),  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=grounded_repair,
    )

    assert repair_calls, "grounded repair callable should have been invoked"
    assert "AttributeError" in repair_calls[0]
    assert impl.read_text() == "GROUNDED FIX\n"


@pytest.mark.asyncio
async def test_apply_targeted_fix_falls_back_to_router_when_repair_is_none(
    tmp_path: Path,
) -> None:
    """`repair=None` keeps the legacy router.complete_structured path (backward compat)."""
    from molexp.agent.modes.author.debug_loop import _apply_targeted_fix
    from molexp.agent.router import ModelTier

    impl = _write(tmp_path / "src" / "experiment" / "tasks" / "run.py", "x = 1\n")
    test = _write(
        tmp_path / "src" / "tests" / "test_run.py",
        "def test() -> None:\n    assert True\n",
    )

    router = _ScriptedFixRouter(["ROUTER FALLBACK FIX\n"])

    await _apply_targeted_fix(
        task_id="run",
        impl_path=impl,
        test_path=test,
        traceback="boom",
        router=router,  # type: ignore[arg-type]
        tier=ModelTier.DEFAULT,
        repair=None,
    )

    assert router.repair_calls == 1
    assert impl.read_text() == "ROUTER FALLBACK FIX\n"
