"""Happy-path tests for the agent_codex plugin.

Subprocess + JSON-RPC dialog is stubbed: each request from the client
unblocks an ``asyncio.Event`` that lets the scripted driver feed the
matching reply onto stdout. Deterministic, no busy-wait.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_plugins.conftest import StreamReader


class _ScriptedStdin:
    """Stdin that records frames and signals waiters by JSON-RPC ``method``.

    ``waiter(method)`` returns an Event that is *already set* if the
    matching frame has already been written, avoiding the registration race
    between the client calling ``write()`` and the test driver calling
    ``waiter()``.
    """

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self._fired_methods: set[str] = set()
        self._waiters: dict[str, asyncio.Event] = {}

    def waiter(self, method: str) -> asyncio.Event:
        event = self._waiters.setdefault(method, asyncio.Event())
        if method in self._fired_methods:
            event.set()
        return event

    def write(self, data: bytes) -> None:
        for raw in data.splitlines():
            if not raw.strip():
                continue
            frame = json.loads(raw.decode("utf-8"))
            self.frames.append(frame)
            method = frame.get("method")
            if method:
                self._fired_methods.add(method)
                if method in self._waiters:
                    self._waiters[method].set()

    async def drain(self) -> None: ...


class _ScriptedStdout:
    """Async-queue-backed stdout stub. Tests push replies via ``feed``."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, payload: dict[str, Any]) -> None:
        self._queue.put_nowait(json.dumps(payload).encode("utf-8") + b"\n")

    async def readline(self) -> bytes:
        return await self._queue.get()


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _ScriptedStdin()
        self.stdout = _ScriptedStdout()
        self.stderr = StreamReader()
        self.pid = 1717
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False

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
    holder: dict[str, Any] = {"proc": None}

    async def _fake_exec(*args, **kwargs):
        proc = _FakeProcess()
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return holder


async def _reply_when(stdin: _ScriptedStdin, method: str) -> dict[str, Any]:
    """Wait for ``method`` to be written to stdin, return its frame."""
    await stdin.waiter(method).wait()
    return next(f for f in stdin.frames if f.get("method") == method)


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

    async def driver() -> None:
        # Wait until start() has spawned the proc and registered _read_stdout.
        while fake_subprocess["proc"] is None:
            await asyncio.sleep(0)
        proc = fake_subprocess["proc"]
        init = await _reply_when(proc.stdin, "initialize")
        proc.stdout.feed({"id": init["id"], "result": {}})
        ts = await _reply_when(proc.stdin, "thread/start")
        proc.stdout.feed({"id": ts["id"], "result": {"thread": {"id": "thr-1"}}})
        tsn = await _reply_when(proc.stdin, "turn/start")
        proc.stdout.feed({"id": tsn["id"], "result": {"turn": {"id": "turn-1"}}})
        proc.stdout.feed(
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}}
        )

    driver_task = asyncio.create_task(driver())

    await client.start()
    thread_id = await client.start_thread()
    assert thread_id == "thr-1"
    result = await client.run_turn(thread_id, "say hi")
    assert result.status == "completed"
    assert result.thread_id == "thr-1"
    await client.close()
    await driver_task
