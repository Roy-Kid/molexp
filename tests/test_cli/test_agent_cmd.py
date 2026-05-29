"""``molexp agent`` REPL — end-to-end via Typer CliRunner (ac-008 / ac-009).

The REPL is driven with a scripted fake router (no live LLM), scripted
stdin (``/help`` → one turn → ``/exit``), and a tmp workspace; the test
asserts rendered output and that the session's ``entries.jsonl`` lands
on disk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.agent.loops import InteractiveLoop
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.types import UsageBreakdown
from molexp.cli import app


class _ScriptedRouter:
    """A fake ``Router`` whose ``stream_agentic`` replays a fixed reply."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[object, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="Hello from ")
        yield TextDeltaChunk(text="the agent.")
        yield FinalChunk(text="Hello from the agent.")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the REPL build an AgentRunner backed by the scripted router."""

    def _fake_make_runner(*, loop: InteractiveLoop, model: str, workspace: Path) -> AgentRunner:
        return AgentRunner(loop=loop, router=_ScriptedRouter(), workspace=workspace, approval=None)

    monkeypatch.setattr("molexp.cli.agent_cmd._make_runner", _fake_make_runner)


@pytest.mark.integration
def test_agent_command_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output


@pytest.mark.integration
def test_agent_repl_runs_a_turn_and_persists_session(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner(monkeypatch)

    result = runner.invoke(
        app,
        ["agent", "--model", "x", "--workspace", str(tmp_path), "--session", "s1"],
        input="/help\nhello there\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    # /help is a REPL-meta command handled in-process
    assert "Commands" in result.output
    # the turn rendered the ModeStarted header + the streamed answer
    assert "interactive" in result.output
    assert "Hello from the agent." in result.output
    # the conversation persisted to the named session
    jsonl = list(tmp_path.rglob("entries.jsonl"))
    assert jsonl, "expected the session's entries.jsonl to be written"


@pytest.mark.integration
def test_agent_repl_exits_cleanly_on_eof(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner(monkeypatch)
    # no /exit — stdin simply runs out
    result = runner.invoke(
        app,
        ["agent", "--model", "x", "--workspace", str(tmp_path)],
        input="hi\n",
    )
    assert result.exit_code == 0, result.output


@pytest.mark.integration
def test_agent_without_model_exits_with_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("molexp.cli.agent_cmd._configured_model", lambda: None)
    result = runner.invoke(app, ["agent", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "No model configured" in result.output
