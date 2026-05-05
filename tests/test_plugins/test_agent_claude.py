"""Happy-path tests for the agent_claude plugin.

The plugin wraps the Claude Code CLI as a coding-agent client. Tests use
a stubbed ``asyncio.create_subprocess_exec`` so no real ``claude`` binary
is required; the stream-json event loop is fed pre-canned lines.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def captured_events() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def on_event(captured_events: list[dict[str, Any]]):
    async def _emit(payload: dict[str, Any]) -> None:
        captured_events.append(payload)

    return _emit


def _stream_json_lines(messages: list[dict[str, Any]]) -> bytes:
    return b"\n".join(json.dumps(m).encode("utf-8") for m in messages) + b"\n"


class _FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._cursor = 0

    async def readline(self) -> bytes:
        if self._cursor >= len(self._payload):
            return b""
        nl = self._payload.find(b"\n", self._cursor)
        if nl == -1:
            chunk = self._payload[self._cursor :]
            self._cursor = len(self._payload)
            return chunk
        chunk = self._payload[self._cursor : nl + 1]
        self._cursor = nl + 1
        return chunk


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None: ...
    def close(self) -> None: ...


class _FakeProcess:
    def __init__(self, stdout: bytes, *, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(b"")
        self._returncode = returncode
        self.pid = 4242
        self.killed = False
        self.terminated = False
        self._wait_result: asyncio.Future[int] | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode if (self.terminated or self.killed) else None

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> int:
        return self._returncode


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Patch ``asyncio.create_subprocess_exec`` to return a scripted fake process."""

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

    stream = _stream_json_lines(
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
    fake_subprocess["proc"] = _FakeProcess(stream)

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
    assert "turn_started" in event_names
    assert "turn_completed" in event_names
    assert "result" in event_names


@pytest.mark.asyncio
async def test_strips_anthropic_env_vars(
    workspace: Path, on_event, fake_subprocess, monkeypatch
):
    from molexp.plugins.agent_claude import ClaudeCliClient, ClaudeCliConfig

    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaked")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "leaked")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "leaked")

    stream = _stream_json_lines(
        [
            {"type": "system", "subtype": "init"},
            {"type": "result", "subtype": "success", "is_error": False},
        ]
    )
    fake_subprocess["proc"] = _FakeProcess(stream)

    client = ClaudeCliClient(
        config=ClaudeCliConfig(strip_anthropic_api_key_env=True),
        workspace=workspace,
        on_event=on_event,
    )
    await client.start()
    thread = await client.start_thread()
    await client.run_turn(thread, "p")

    env_used = fake_subprocess["env"]
    assert "ANTHROPIC_API_KEY" not in env_used
    assert "ANTHROPIC_AUTH_TOKEN" not in env_used
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env_used


@pytest.mark.asyncio
async def test_thread_id_mismatch_raises(workspace: Path, on_event, fake_subprocess):
    from molexp.plugins.agent_claude import ClaudeCliClient, ClaudeCliConfig
    from molexp.plugins.coding_agent import AgentError

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
