"""``ChatMode`` unit tests (spec ac-008)."""

from __future__ import annotations

import pytest

from molexp.agent.modes import ChatMode, ChatModeConfig


def test_chat_mode_carries_config() -> None:
    cfg = ChatModeConfig(system_prompt="you are helpful")
    mode = ChatMode(config=cfg)
    assert mode.name == "chat"
    assert mode.config is cfg


def test_chat_mode_config_is_frozen() -> None:
    cfg = ChatModeConfig(system_prompt="x")
    with pytest.raises(Exception):
        cfg.system_prompt = "y"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_chat_mode_run_returns_non_empty_text_via_test_model() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.router import PydanticAIRouter
    from molexp.agent.mode import AgentRunResult
    from molexp.agent.router import ModelTier
    from molexp.agent.session import AgentSession

    test_model = TestModel()
    router = PydanticAIRouter(
        models={
            ModelTier.CHEAP: test_model,
            ModelTier.DEFAULT: test_model,
            ModelTier.HEAVY: test_model,
        },
    )
    mode = ChatMode(config=ChatModeConfig())
    result = await mode.run(router=router, session=AgentSession(), user_input="ping")
    assert isinstance(result, AgentRunResult)
    assert result.text
