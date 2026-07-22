"""``PydanticAIRouter`` forwards both pydantic-ai-native tool shapes verbatim.

``AgentRunner.tools`` / ``PydanticAIRouter.tools`` are passed straight into
``pydantic_ai.Agent(tools=[...])`` with no molexp middle layer. The SDK natively
accepts a :class:`pydantic_ai.tools.Tool` instance or a bare callable; both are
forwarded on the text path. Asserted by spying on the ``Agent`` constructor.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.tools import Tool

from molexp.agent._pydanticai.router import PydanticAIRouter
from molexp.agent.router import ModelTier


def _models_all(model: object) -> dict[ModelTier, object]:
    return dict.fromkeys(ModelTier, model)


class _AgentSpyResult:
    """Minimal stand-in for pydantic-ai's ``RunResult``."""

    output = "ok"

    @property
    def usage(self) -> object:
        class _U:
            input_tokens = 0
            output_tokens = 0
            cache_read_tokens = 0
            cache_write_tokens = 0
            total_tokens = 0
            requests = 1

        return _U()


class _AgentSpy:
    """Captures the constructor kwargs of the patched ``Agent``."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    async def run(self, user: str, message_history: object | None = None) -> _AgentSpyResult:
        del user, message_history
        return _AgentSpyResult()


@pytest.fixture(autouse=True)
def _reset_spy() -> None:
    _AgentSpy.last_kwargs = None


class TestToolForwarding:
    @pytest.mark.asyncio
    async def test_mixed_tool_shapes_forwarded_as_list_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tuple mixing a ``Tool`` and a bare callable forwards as an ordered list."""

        async def greet(name: str) -> str:
            return f"hi {name}"

        async def echo(message: str) -> str:
            return message

        tool = Tool(greet)

        monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
        router = PydanticAIRouter(models=_models_all("x"), tools=(tool, echo))
        await router.complete_text(prompt="hi")

        captured = _AgentSpy.last_kwargs
        assert captured is not None
        assert captured["tools"] == [tool, echo]

    @pytest.mark.asyncio
    async def test_empty_tools_omits_tools_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No tools → ``Agent`` is built without a ``tools=`` kwarg, never an empty list."""
        monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
        router = PydanticAIRouter(models=_models_all("x"))
        await router.complete_text(prompt="hi")

        captured = _AgentSpy.last_kwargs
        assert captured is not None
        assert "tools" not in captured
