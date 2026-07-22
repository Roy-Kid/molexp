"""Tests for the unified :class:`PydanticAIRouter`.

Drives the router against an in-memory stub of ``pydantic_ai.Agent``
(pre-populated into the router's ``(tier, schema|None)`` cache so the real SDK
``Agent`` is never constructed). Covers construction validation, the structured
and text paths, the shared transport-retry shim, provider-event emission, hook
resilience, and :class:`Router` protocol conformance.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from molexp.agent._pydanticai.errors import ErrorKind, ProviderError
from molexp.agent._pydanticai.events import Outcome, ProviderEvent
from molexp.agent._pydanticai.retry import RetryPolicy
from molexp.agent._pydanticai.router import PydanticAIRouter
from molexp.agent.router import ModelTier, Router, RouterTextResult


class _Out(BaseModel):
    """Simple structured-output schema used in tests."""

    payload: str


class _StubAgentResult:
    """Mimics the ``.output``-bearing object ``pydantic_ai.Agent.run`` returns."""

    def __init__(self, output: object) -> None:
        self.output = output


class _StubAgent:
    """In-memory replacement for ``pydantic_ai.Agent``.

    Each :meth:`run` consumes the next behavior from ``script``: a
    :class:`BaseException` instance is raised, anything else is wrapped in
    :class:`_StubAgentResult` and returned. Prompts are recorded for assertion.
    """

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls: list[str] = []

    async def run(self, user: str, message_history: object | None = None) -> _StubAgentResult:
        del message_history  # not asserted on by these tests
        self.calls.append(user)
        if not self._script:
            raise RuntimeError("stub script exhausted")
        nxt = self._script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return _StubAgentResult(nxt)


def _models_all(model: object) -> dict[ModelTier, object]:
    return {ModelTier.CHEAP: model, ModelTier.DEFAULT: model, ModelTier.HEAVY: model}


def _install_structured_stub(
    router: PydanticAIRouter, tier: ModelTier, schema: type[BaseModel], stub: _StubAgent
) -> None:
    """Bypass ``_structured_agent`` by pre-populating the cache."""
    router._agents[(tier, schema)] = stub  # type: ignore[assignment]


def _install_text_stub(router: PydanticAIRouter, tier: ModelTier, stub: _StubAgent) -> None:
    """Bypass ``_text_agent`` by pre-populating the cache."""
    router._agents[(tier, None)] = stub  # type: ignore[assignment]


class TestPydanticAIRouter:
    def test_construction_requires_all_tiers(self) -> None:
        with pytest.raises(ValueError, match="must cover every ModelTier"):
            PydanticAIRouter(models={ModelTier.DEFAULT: "x"})

    @pytest.mark.asyncio
    async def test_complete_structured_happy_path_emits_ok_events(self) -> None:
        starts: list[ProviderEvent] = []
        ends: list[ProviderEvent] = []
        router = PydanticAIRouter(
            models=_models_all("x"),
            on_invoke_start=starts.append,
            on_invoke_end=ends.append,
        )
        stub = _StubAgent([_Out(payload="ok")])
        _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

        result = await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hello", schema=_Out, node_id="ingest"
        )
        assert result == _Out(payload="ok")
        assert stub.calls == ["hello"]
        assert [s.outcome for s in starts] == [Outcome.ok]
        assert starts[0].attempt == 1
        assert [e.outcome for e in ends] == [Outcome.ok]
        assert ends[0].attempt == 1

    @pytest.mark.asyncio
    async def test_complete_structured_retries_transient_failures_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

        starts: list[ProviderEvent] = []
        ends: list[ProviderEvent] = []
        router = PydanticAIRouter(
            models=_models_all("x"),
            retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.1),
            on_invoke_start=starts.append,
            on_invoke_end=ends.append,
        )
        stub = _StubAgent([TimeoutError(), TimeoutError(), _Out(payload="finally")])
        _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

        result = await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
        assert result == _Out(payload="finally")
        assert [s.attempt for s in starts] == [1, 2, 3]
        assert [e.outcome for e in ends] == [Outcome.retry, Outcome.retry, Outcome.ok]
        assert len(sleeps) == 2

    @pytest.mark.asyncio
    async def test_complete_structured_retry_exhaustion_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _fake_sleep(seconds: float) -> None:
            del seconds

        monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

        underlying = ConnectionError("refused")
        router = PydanticAIRouter(
            models=_models_all("x"),
            retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
        )
        stub = _StubAgent([underlying, underlying, underlying])
        _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

        with pytest.raises(ProviderError) as exc_info:
            await router.complete_structured(
                tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out, node_id="ingest"
            )
        err = exc_info.value
        assert err.kind is ErrorKind.model_unavailable
        assert err.attempts == 3
        assert err.node_id == "ingest"
        assert err.tier is ModelTier.DEFAULT
        assert err.cause is underlying

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_break_invoke(self) -> None:
        """A faulty telemetry sink must not poison the LLM call path: the
        router catches and logs the hook exception and still returns the
        schema-typed result."""

        def _bad_end(_event: ProviderEvent) -> None:
            raise RuntimeError("telemetry sink failed")

        router = PydanticAIRouter(models=_models_all("x"), on_invoke_end=_bad_end)
        stub = _StubAgent([_Out(payload="ok")])
        _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

        result = await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
        assert result == _Out(payload="ok")

    @pytest.mark.asyncio
    async def test_complete_text_returns_router_text_result(self) -> None:
        router = PydanticAIRouter(models=_models_all("x"))
        stub = _StubAgent(["hello back"])
        _install_text_stub(router, ModelTier.DEFAULT, stub)

        result = await router.complete_text(prompt="hi")
        assert isinstance(result, RouterTextResult)
        assert result.text == "hello back"
        assert stub.calls == ["hi"]

    @pytest.mark.asyncio
    async def test_complete_text_recovers_from_transport_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ac-001 — the text path must recover from a first-attempt transport
        failure exactly as the structured path does (both share
        ``_run_with_transport_retry``). ``TimeoutError`` classifies as
        ``timeout``, which is retryable, so attempt #2 succeeds."""

        async def _fake_sleep(seconds: float) -> None:
            del seconds

        monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

        router = PydanticAIRouter(
            models=_models_all("x"),
            retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
        )
        stub = _StubAgent([TimeoutError(), "recovered"])
        _install_text_stub(router, ModelTier.DEFAULT, stub)

        result = await router.complete_text(prompt="hi")
        assert result.text == "recovered"
        assert stub.calls == ["hi", "hi"]

    @pytest.mark.asyncio
    async def test_schema_parse_not_retried_by_router_loop(self) -> None:
        """ac-004 — prod-incident lock. A wrong-typed structured output makes
        the router raise ``TypeError`` (→ ``schema_parse``), which is absent
        from the default ``retry_on``: exactly one ``run()`` call, then a
        terminal ``ProviderError(kind=schema_parse, attempts=1)``."""
        router = PydanticAIRouter(models=_models_all("x"))
        stub = _StubAgent(["a bare string, not an _Out instance"])
        _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

        with pytest.raises(ProviderError) as exc_info:
            await router.complete_structured(
                tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
            )
        assert exc_info.value.kind is ErrorKind.schema_parse
        assert exc_info.value.attempts == 1
        assert stub.calls == ["hi"]

    def test_conforms_to_router_protocol(self) -> None:
        """The sanctioned harness→agent edge depends on this concrete impl
        satisfying the :class:`Router` protocol structurally."""
        assert isinstance(PydanticAIRouter(models=_models_all("x")), Router)
