"""Tests for the shared coding-agent plugin contract.

Covers the provider-neutral types (``CodingAgentClient`` Protocol, ``TurnResult``
dataclass, ``AgentError`` exception, ``AgentEventCallback`` alias) that
``agent_claude`` and ``agent_codex`` plugins implement. These tests are
intentionally light — they pin the public surface, not implementation
details. Each plugin has its own integration tests.
"""

from __future__ import annotations

from typing import Any, get_type_hints

import pytest


def test_coding_agent_module_importable():
    import molexp.plugins.coding_agent as ca

    assert ca is not None


def test_turn_result_dataclass_shape():
    from molexp.plugins.coding_agent import TurnResult

    result = TurnResult(thread_id="t1", turn_id="u1", status="completed")
    assert result.thread_id == "t1"
    assert result.turn_id == "u1"
    assert result.status == "completed"


def test_agent_error_is_runtime_error():
    from molexp.plugins.coding_agent import AgentError

    assert issubclass(AgentError, RuntimeError)
    err = AgentError("nope")
    assert str(err) == "nope"


def test_agent_event_callback_alias_exists():
    from molexp.plugins.coding_agent import AgentEventCallback

    # AgentEventCallback is a typing alias; merely importable + non-None is fine.
    assert AgentEventCallback is not None


def test_coding_agent_client_protocol_is_runtime_checkable():
    from molexp.plugins.coding_agent import CodingAgentClient

    # CodingAgentClient must be a runtime-checkable Protocol so plugins can
    # be type-checked dynamically.
    class _Impl:
        pid: int | None = None

        async def start(self) -> None: ...
        async def start_thread(self) -> str:
            return ""
        async def run_turn(self, thread_id: str, prompt: str):
            from molexp.plugins.coding_agent import TurnResult

            return TurnResult(thread_id="", turn_id="", status="")
        async def close(self) -> None: ...

    assert isinstance(_Impl(), CodingAgentClient)


def test_protocol_rejects_missing_methods():
    from molexp.plugins.coding_agent import CodingAgentClient

    class _Partial:
        pid: int | None = None

        async def start(self) -> None: ...
        # missing start_thread / run_turn / close

    assert not isinstance(_Partial(), CodingAgentClient)
