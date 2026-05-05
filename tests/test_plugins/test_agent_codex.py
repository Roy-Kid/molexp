"""Happy-path tests for the agent_codex plugin.

The plugin wraps the Codex app-server (``codex app-server``) as a coding-agent
client. It exchanges JSON-RPC over stdio with the long-lived subprocess.
Tests stub ``asyncio.create_subprocess_exec`` so no real binary is required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def captured_events() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def on_event(captured_events: list[dict[str, Any]]):
    async def _emit(payload: dict[str, Any]) -> None:
        captured_events.append(payload)

    return _emit


class _ScriptedStdout:
    """Stdout that emits queued JSON-RPC frames in order."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False

    def feed(self, payload: dict[str, Any]) -> None:
        self._queue.put_nowait(json.dumps(payload).encode("utf-8") + b"\n")

    def feed_eof(self) -> None:
        self._closed = True
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


class _NullStream:
    async def readline(self) -> bytes:
        return b""


class _CapturedStdin:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def write(self, data: bytes) -> None:
        for raw in data.splitlines():
            if raw.strip():
                self._sink.append(json.loads(raw.decode("utf-8")))

    async def drain(self) -> None: ...


class _FakeProcess:
    def __init__(self, stdin_sink: list[dict[str, Any]]) -> None:
        self.stdin = _CapturedStdin(stdin_sink)
        self.stdout = _ScriptedStdout()
        self.stderr = _NullStream()
        self.pid = 1717
        self.terminated = False
        self.killed = False
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    async def wait(self) -> int:
        return self._returncode or 0


@pytest.fixture
def fake_subprocess(monkeypatch):
    holder: dict[str, Any] = {"proc": None, "stdin_sink": []}

    async def _fake_exec(*args, **kwargs):
        proc = _FakeProcess(holder["stdin_sink"])
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return holder


@pytest.mark.asyncio
async def test_run_turn_happy_path(
    tmp_path: Path,
    on_event,
    captured_events,
    fake_subprocess,
):
    from molexp.plugins.agent_codex import CodexAppServerClient, CodexConfig

    client = CodexAppServerClient(
        config=CodexConfig(command="echo codex", read_timeout_ms=1000, turn_timeout_ms=1000),
        workspace=tmp_path,
        on_event=on_event,
    )

    async def _scripted_responses() -> None:
        # Wait for the proc to exist (start() spawns it).
        for _ in range(50):
            await asyncio.sleep(0.01)
            if fake_subprocess["proc"] is not None:
                break
        proc = fake_subprocess["proc"]
        # Drive the JSON-RPC dialog: respond to each request as it arrives.
        sink = fake_subprocess["stdin_sink"]
        # initialize
        for _ in range(50):
            if any(r.get("method") == "initialize" for r in sink):
                break
            await asyncio.sleep(0.01)
        init = next(r for r in sink if r.get("method") == "initialize")
        proc.stdout.feed({"id": init["id"], "result": {}})
        # thread/start
        for _ in range(50):
            if any(r.get("method") == "thread/start" for r in sink):
                break
            await asyncio.sleep(0.01)
        ts = next(r for r in sink if r.get("method") == "thread/start")
        proc.stdout.feed({"id": ts["id"], "result": {"thread": {"id": "thr-1"}}})
        # turn/start
        for _ in range(50):
            if any(r.get("method") == "turn/start" for r in sink):
                break
            await asyncio.sleep(0.01)
        tsn = next(r for r in sink if r.get("method") == "turn/start")
        proc.stdout.feed({"id": tsn["id"], "result": {"turn": {"id": "turn-1"}}})
        # turn/completed notification
        proc.stdout.feed({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})

    driver = asyncio.create_task(_scripted_responses())

    await client.start()
    thread_id = await client.start_thread()
    assert thread_id == "thr-1"
    result = await client.run_turn(thread_id, "say hi")
    assert result.status == "completed"
    assert result.thread_id == "thr-1"
    await client.close()
    await driver
