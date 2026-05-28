"""Shared fixtures for the agent-harness test suite.

Provides the three test doubles the spec calls for: a
``FakeExecutionEnv`` (no real subprocesses), a scripted ``Router`` stub
(canned text/structured completions), and convenient ``Session`` /
``InMemorySessionStorage`` builders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from molexp.agent.execution_env import ExecResult
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown


class FakeExecutionEnv:
    """In-memory :class:`~molexp.agent.execution_env.ExecutionEnv`.

    Records every ``exec`` call and returns a canned result; spawns no
    real process. Satisfies the ``ExecutionEnv`` Protocol structurally.
    """

    def __init__(self, *, scratch_dir: Path, result: ExecResult | None = None) -> None:
        self._scratch_dir = scratch_dir
        self._result = result or ExecResult(stdout="fake-ok", stderr="", exit_code=0)
        self.calls: list[dict[str, object]] = []

    @property
    def scratch_dir(self) -> Path:
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        return self._scratch_dir

    def exec(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "env": dict(env) if env is not None else None,
                "timeout": timeout,
            }
        )
        return self._result


class ScriptedRouter:
    """A :class:`~molexp.agent.router.Router` stub returning canned text.

    ``complete_text`` pops the next scripted response (or echoes the
    prompt when the script is exhausted); every call is recorded.
    ``complete_structured`` is unused by the harness and asserts if hit.
    """

    def __init__(self, responses: Sequence[str] = ()) -> None:
        self._responses: list[str] = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "history": tuple(message_history),
                "tier": tier,
            }
        )
        if self._responses:
            return RouterTextResult(text=self._responses.pop(0))
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("the harness never calls complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


@pytest.fixture
def in_memory_storage() -> InMemorySessionStorage:
    return InMemorySessionStorage()


@pytest.fixture
def session(in_memory_storage: InMemorySessionStorage) -> Session:
    return Session(storage=in_memory_storage, session_id="test-session")


@pytest.fixture
def scripted_router() -> ScriptedRouter:
    return ScriptedRouter(responses=["canned summary"])


@pytest.fixture
def fake_execution_env(tmp_path: Path) -> FakeExecutionEnv:
    return FakeExecutionEnv(scratch_dir=tmp_path / "scratch")
