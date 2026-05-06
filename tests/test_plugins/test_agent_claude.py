"""Happy-path tests for the agent_claude plugin.

Subprocess is stubbed via ``asyncio.create_subprocess_exec`` monkey-patch;
no real ``claude`` binary is required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.test_plugins.conftest import CapturedStdin, StreamReader, stream_json_lines


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


class _FakeProcess:
    def __init__(self, stdout: bytes) -> None:
        self.stdin = CapturedStdin()
        self.stdout = StreamReader(stdout)
        self.stderr = StreamReader()
        self.pid = 4242
        self.killed = False
        self.terminated = False
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode if (self.terminated or self.killed) else None

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    async def wait(self) -> int:
        return self._returncode or 0


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Patch subprocess spawn to return a scripted fake process."""
    holder: dict[str, Any] = {"proc": None, "args": None, "env": None}

    async def _fake_exec(*args, **kwargs):
        proc = holder["proc"]
        if proc is None:
            raise RuntimeError("test forgot to set holder['proc']")
        holder["args"] = args
        holder["env"] = kwargs.get("env")
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        "shutil.which", lambda binary: "/usr/bin/claude" if binary == "claude" else None
    )
    return holder


@pytest.mark.asyncio
async def test_run_turn_happy_path(
    workspace: Path,
    on_event,
    captured_events: list[dict[str, Any]],
    fake_subprocess,
):
    from molexp.plugins.agent_claude import ClaudeCliClient, ClaudeCliConfig

    fake_subprocess["proc"] = _FakeProcess(
        stream_json_lines(
            [
                {"type": "system", "subtype": "init", "session_id": "s-1", "model": "haiku"},
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "hello"}]},
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "duration_ms": 12,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ]
        )
    )

    client = ClaudeCliClient(
        config=ClaudeCliConfig(),
        workspace=workspace,
        on_event=on_event,
    )
    await client.start()
    thread_id = await client.start_thread()
    result = await client.run_turn(thread_id, "say hi")
    await client.close()

    assert result.status == "completed"
    assert result.thread_id == thread_id
    event_names = {ev["event"] for ev in captured_events}
    assert {"turn_started", "turn_completed", "result"} <= event_names


@pytest.mark.asyncio
async def test_strips_anthropic_env_vars(workspace: Path, on_event, fake_subprocess, monkeypatch):
    from molexp.plugins.agent_claude import ClaudeCliClient, ClaudeCliConfig

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.setenv(name, "leaked")

    fake_subprocess["proc"] = _FakeProcess(
        stream_json_lines(
            [
                {"type": "system", "subtype": "init"},
                {"type": "result", "subtype": "success", "is_error": False},
            ]
        )
    )

    client = ClaudeCliClient(
        config=ClaudeCliConfig(strip_anthropic_api_key_env=True),
        workspace=workspace,
        on_event=on_event,
    )
    await client.start()
    thread = await client.start_thread()
    await client.run_turn(thread, "p")

    env_used = fake_subprocess["env"]
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert name not in env_used


@pytest.mark.asyncio
async def test_thread_id_mismatch_raises(workspace: Path, on_event, fake_subprocess):
    from molexp.plugins.agent_claude import ClaudeCliClient, ClaudeCliConfig
    from molexp.agent.coding_protocol import AgentError

    fake_subprocess["proc"] = _FakeProcess(b"")
    client = ClaudeCliClient(
        config=ClaudeCliConfig(),
        workspace=workspace,
        on_event=on_event,
    )
    await client.start()
    await client.start_thread()
    with pytest.raises(AgentError):
        await client.run_turn("wrong-thread", "p")
