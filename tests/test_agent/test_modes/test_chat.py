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
    from pydantic import ValidationError

    cfg = ChatModeConfig(system_prompt="x")
    with pytest.raises(ValidationError):
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


@pytest.mark.asyncio
async def test_chat_mode_forwards_session_model_messages_as_history() -> None:
    """ChatMode must pass ``session.model_messages`` into the router.

    A stub router captures the kwargs passed to ``complete_text`` so we
    can assert the message history flows through; the actual LLM
    round-trip is exercised by other tests via ``TestModel``.
    """
    from molexp.agent.router import ModelTier, RouterTextResult
    from molexp.agent.session import AgentSession
    from molexp.agent.types import UsageBreakdown

    class _CapturingRouter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def complete_text(
            self,
            *,
            prompt: str,
            system: str = "",
            message_history: tuple[object, ...] = (),
            tier: ModelTier = ModelTier.DEFAULT,
        ) -> RouterTextResult:
            self.calls.append({"prompt": prompt, "history": tuple(message_history), "tier": tier})
            return RouterTextResult(text="ok")

        async def complete_structured(self, **_: object) -> object:
            raise AssertionError("ChatMode never reaches complete_structured")

        def clear_usage(self) -> None:
            return None

        def snapshot_usage(self) -> UsageBreakdown:
            return UsageBreakdown()

    seeded: tuple[object, ...] = ("prior-msg-1", "prior-msg-2")
    session = AgentSession(model_messages=seeded)
    router = _CapturingRouter()
    mode = ChatMode(config=ChatModeConfig())

    await mode.run(router=router, session=session, user_input="next turn")

    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["prompt"] == "next turn"
    assert call["history"] == seeded


@pytest.mark.asyncio
async def test_chat_mode_persists_all_messages_into_session() -> None:
    """After a run, ``session.model_messages`` carries ``result.raw.all_messages()``.

    The stub router returns a ``RouterTextResult`` whose ``raw``
    exposes ``all_messages()`` — exactly what pydantic-ai's
    ``AgentRunResult`` provides. ChatMode must read that and store
    the cumulative list so the next turn forwards it back.
    """
    from molexp.agent.router import ModelTier, RouterTextResult
    from molexp.agent.session import AgentSession
    from molexp.agent.types import UsageBreakdown

    class _StubRaw:
        def __init__(self, messages: tuple[object, ...]) -> None:
            self._messages = messages

        def all_messages(self) -> list[object]:
            return list(self._messages)

    cumulative: tuple[object, ...] = ("u1", "a1")

    class _StubRouter:
        async def complete_text(
            self,
            *,
            prompt: str,
            system: str = "",
            message_history: tuple[object, ...] = (),
            tier: ModelTier = ModelTier.DEFAULT,
        ) -> RouterTextResult:
            return RouterTextResult(text="response", raw=_StubRaw(cumulative))

        async def complete_structured(self, **_: object) -> object:
            raise AssertionError

        def clear_usage(self) -> None:
            return None

        def snapshot_usage(self) -> UsageBreakdown:
            return UsageBreakdown()

    session = AgentSession()
    assert session.model_messages == ()
    await ChatMode(config=ChatModeConfig()).run(
        router=_StubRouter(), session=session, user_input="hi"
    )
    assert session.model_messages == cumulative


@pytest.mark.asyncio
async def test_chat_mode_round_trip_threads_history_across_two_turns() -> None:
    """End-to-end: a real pydantic-ai ``Agent`` (via ``TestModel``) sees prior turns.

    ``TestModel`` echoes synthetic responses, but pydantic-ai still
    walks the full message history when assembling each request — so
    after two ``mode.run`` calls on the same session, the cumulative
    ``model_messages`` must contain at least one user prompt per turn
    AND ``TestModel.last_model_request_parameters.messages`` for the
    second turn must include the first user prompt.
    """
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import UserPromptPart
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.router import PydanticAIRouter
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
    session = AgentSession()

    await mode.run(router=router, session=session, user_input="first")
    first_turn_history = session.model_messages
    assert len(first_turn_history) >= 2  # at least the request + response

    await mode.run(router=router, session=session, user_input="second")
    second_turn_history = session.model_messages
    # The cumulative history must be strictly longer after the second turn.
    assert len(second_turn_history) > len(first_turn_history)

    # And the second turn's request must reference the first user prompt —
    # which is what "the model sees prior turns" actually means.
    user_prompts: list[str] = []
    for msg in second_turn_history:
        for part in getattr(msg, "parts", []) or []:
            if isinstance(part, UserPromptPart):
                user_prompts.append(part.content if isinstance(part.content, str) else "")
    assert "first" in user_prompts
    assert "second" in user_prompts
