"""``AgentRunner`` end-to-end (spec ac-008 / ac-012)."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from molexp.agent.mode import AgentRunResult
from molexp.agent.modes import ChatMode, ChatModeConfig
from molexp.agent.runner import AgentRunner
from molexp.agent.session import AgentSession


def test_construction_no_network_io() -> None:
    """ac-012 — instantiation must not touch the network."""

    real_socket = socket.socket

    def deny(*args, **kwargs):
        raise AssertionError("AgentRunner construction touched the network")

    with patch("socket.socket", side_effect=deny):
        runner = AgentRunner(
            mode=ChatMode(config=ChatModeConfig()),
            model="openai:gpt-5.2",
        )
    # Restore for any teardown noise; harness still unconstructed at this point.
    socket.socket = real_socket
    assert runner.mode is not None
    assert runner.model == "openai:gpt-5.2"


@pytest.mark.asyncio
async def test_chat_mode_round_trip() -> None:
    """ac-008 — ChatMode + TestModel returns non-empty AgentRunResult."""

    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    runner = AgentRunner(
        mode=ChatMode(config=ChatModeConfig()),
        model=TestModel(),  # type: ignore[arg-type]
    )
    session = AgentSession()
    result = await runner.run(session, "hello")
    assert isinstance(result, AgentRunResult)
    assert result.text
