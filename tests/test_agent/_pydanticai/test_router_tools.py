"""Verify the router accepts both pydantic-ai-native tool shapes.

Phase 2 of ``agent-pydanticai-rectification`` collapses the tool
injection path so that ``AgentRunner.tools`` and
``PydanticAIRouter.tools`` are passed verbatim into
``pydantic_ai.Agent(tools=[...])`` with no molexp middle layer. The
SDK natively accepts two shapes:

* a :class:`pydantic_ai.tools.Tool` instance — built via the ``Tool``
  decorator/factory;
* a bare callable — pydantic-ai introspects the signature on
  construction.

These tests assert both shapes are forwarded into ``Agent(tools=...)``
on the text path, by spying on the ``Agent`` constructor.
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


@pytest.mark.asyncio
async def test_tool_instance_is_forwarded_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A :class:`pydantic_ai.tools.Tool` instance survives the round-trip."""

    async def greet(name: str) -> str:
        return f"hi {name}"

    tool = Tool(greet)

    monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
    router = PydanticAIRouter(models=_models_all("x"), tools=(tool,))
    await router.complete_text(prompt="hi")

    captured = _AgentSpy.last_kwargs
    assert captured is not None, "Agent was never constructed"
    assert "tools" in captured, "tools= was not forwarded to Agent"
    assert captured["tools"] == [tool]
    assert isinstance(captured["tools"][0], Tool)


@pytest.mark.asyncio
async def test_bare_callable_is_forwarded_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare async callable is also accepted; pydantic-ai wraps it itself."""

    async def echo(message: str) -> str:
        return message

    monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
    router = PydanticAIRouter(models=_models_all("x"), tools=(echo,))
    await router.complete_text(prompt="hi")

    captured = _AgentSpy.last_kwargs
    assert captured is not None, "Agent was never constructed"
    assert "tools" in captured, "tools= was not forwarded to Agent"
    assert captured["tools"] == [echo]
    # Crucially: the router does not wrap the callable in a molexp
    # middle layer — pydantic-ai gets the raw function.
    assert callable(captured["tools"][0])
    assert not isinstance(captured["tools"][0], Tool)


@pytest.mark.asyncio
async def test_mixed_shapes_are_forwarded_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tuple mixing both shapes is forwarded as a list, preserving order."""

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
async def test_empty_tools_omits_tools_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tools → ``Agent`` is built without a ``tools=`` kwarg.

    Avoids handing pydantic-ai an empty list when the user did not
    register any tools — keeps the construction call shape identical to
    the pre-rewrite text-only path.
    """
    monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
    router = PydanticAIRouter(models=_models_all("x"))
    await router.complete_text(prompt="hi")

    captured = _AgentSpy.last_kwargs
    assert captured is not None
    assert "tools" not in captured
