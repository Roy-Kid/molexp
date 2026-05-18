"""Shared fixtures for ``test_plugins/`` — coding-agent provider tests.

Both Claude and Codex client tests need a captured event sink and a fake
subprocess; rather than redefine those per test module, they live here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

# ── event capture ──────────────────────────────────────────────────────────


@pytest.fixture
def captured_events() -> list[dict[str, Any]]:
    """Append-only sink for normalized provider events."""
    return []


@pytest.fixture
def on_event(captured_events: list[dict[str, Any]]):
    """Async callback that appends each emitted payload to ``captured_events``."""

    async def _emit(payload: dict[str, Any]) -> None:
        captured_events.append(payload)

    return _emit


# ── shared subprocess fakes ────────────────────────────────────────────────


class StreamReader:
    """Stdout/stderr stub backed by a static byte buffer split on newlines."""

    def __init__(self, payload: bytes = b"") -> None:
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


class CapturedStdin:
    """Stdin stub. ``write`` accumulates raw bytes and decoded JSON-RPC frames."""

    def __init__(self) -> None:
        self.written = b""
        self.frames: list[dict[str, Any]] = []

    def write(self, data: bytes) -> None:
        self.written += data
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:  # noqa: SIM105
                self.frames.append(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                # Non-JSON writes (e.g. raw prompts to claude CLI) are ignored
                # for the JSON-RPC use case but kept in ``written``.
                pass

    async def drain(self) -> None: ...

    def close(self) -> None: ...


def stream_json_lines(messages: list[dict[str, Any]]) -> bytes:
    """Encode messages as newline-delimited JSON for stdout stubs."""
    return b"\n".join(json.dumps(m).encode("utf-8") for m in messages) + b"\n"
