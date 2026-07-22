"""``PydanticAIRouter.stream_agentic`` forwards ``toolsets=`` into ``Agent``."""

from __future__ import annotations

from typing import Any

import pytest

from molexp.agent._pydanticai.router import PydanticAIRouter
from molexp.agent.router import FinalChunk, ModelTier


def _models_all(model: object) -> dict[ModelTier, object]:
    return dict.fromkeys(ModelTier, model)


class _Usage:
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    total_tokens = 0
    requests = 1


class _RunResult:
    output = "done"
    usage = _Usage()

    def all_messages_json(self) -> bytes:
        return b"[]"


class _EmptyRun:
    result = _RunResult()

    def __aiter__(self) -> _EmptyRun:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


class _IterCM:
    async def __aenter__(self) -> _EmptyRun:
        return _EmptyRun()

    async def __aexit__(self, *exc: object) -> None:
        return None


class _AgentSpy:
    """Captures constructor kwargs; ``iter`` yields no nodes."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def iter(self, prompt: str, message_history: object | None = None) -> _IterCM:
        del prompt, message_history
        return _IterCM()


@pytest.fixture(autouse=True)
def _reset_spy() -> None:
    _AgentSpy.last_kwargs = None


class TestStreamAgenticToolsets:
    @pytest.mark.asyncio
    async def test_toolsets_forwarded_as_list_and_final_carries_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
        router = PydanticAIRouter(models=_models_all("x"))
        fake_toolset = object()

        chunks = [
            c
            async for c in router.stream_agentic(
                prompt="hi",
                toolsets=(fake_toolset,),
            )
        ]

        captured = _AgentSpy.last_kwargs
        assert captured is not None
        assert captured["toolsets"] == [fake_toolset]
        assert isinstance(chunks[-1], FinalChunk)
        assert chunks[-1].text == "done"

    @pytest.mark.asyncio
    async def test_empty_toolsets_omits_toolsets_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("molexp.agent._pydanticai.router.Agent", _AgentSpy)
        router = PydanticAIRouter(models=_models_all("x"))

        _ = [c async for c in router.stream_agentic(prompt="hi")]

        captured = _AgentSpy.last_kwargs
        assert captured is not None
        assert "toolsets" not in captured
