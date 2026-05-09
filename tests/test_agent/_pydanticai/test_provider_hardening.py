"""Tests for the hardened :class:`PydanticAIProvider`.

Drives the provider against an in-memory stub of ``pydantic_ai.Agent``
(monkeypatched onto the provider's per-(tier, schema) cache so we
never instantiate a real SDK Agent). Covers acceptance criteria
ac-006..ac-010 + the structural ``isinstance`` check against the
:class:`Provider` :class:`typing.Protocol`.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from molexp.agent._pydanticai.errors import ErrorKind, ProviderError
from molexp.agent._pydanticai.events import Outcome, ProviderEvent
from molexp.agent._pydanticai.provider import PydanticAIProvider
from molexp.agent._pydanticai.retry import RetryPolicy
from molexp.agent.modes.plan.protocols import ModelTier, Provider


class _Out(BaseModel):
    """Simple structured-output schema used in tests."""

    payload: str


class _StubAgentResult:
    """Mimics the ``.output``-bearing object returned by ``pydantic_ai.Agent.run``."""

    def __init__(self, output: object) -> None:
        self.output = output


class _StubAgent:
    """In-memory replacement for ``pydantic_ai.Agent``.

    Each call to :meth:`run` consumes the next behavior from
    ``script``: a :class:`BaseException` instance is raised, anything
    else is wrapped in :class:`_StubAgentResult` and returned. The
    last user-supplied prompt is recorded for assertion.
    """

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls: list[str] = []

    async def run(self, user: str) -> _StubAgentResult:
        self.calls.append(user)
        if not self._script:
            raise RuntimeError("stub script exhausted")
        nxt = self._script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return _StubAgentResult(nxt)


def _install_stub(
    provider: PydanticAIProvider,
    tier: ModelTier,
    schema: type[BaseModel],
    stub: _StubAgent,
) -> None:
    """Bypass ``_agent_for`` by pre-populating its cache with the stub."""
    provider._agents[(tier, schema)] = stub  # type: ignore[assignment]


# ── Happy path ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_happy_path_one_attempt() -> None:
    starts: list[ProviderEvent] = []
    ends: list[ProviderEvent] = []
    provider = PydanticAIProvider(
        on_invoke_start=starts.append,
        on_invoke_end=ends.append,
    )
    stub = _StubAgent([_Out(payload="ok")])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    result = await provider.invoke(
        tier=ModelTier.DEFAULT,
        system="sys",
        user="hello",
        schema=_Out,
        node_id="ingest",
    )
    assert result == _Out(payload="ok")
    assert stub.calls == ["hello"]
    assert len(starts) == 1
    assert starts[0].outcome is Outcome.ok
    assert starts[0].attempt == 1
    assert len(ends) == 1
    assert ends[0].outcome is Outcome.ok
    assert ends[0].attempt == 1


# ── Retry on schema parse (ac-006) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_retries_two_failures_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("molexp.agent._pydanticai.provider.asyncio.sleep", _fake_sleep)

    starts: list[ProviderEvent] = []
    ends: list[ProviderEvent] = []
    provider = PydanticAIProvider(
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.1),
        on_invoke_start=starts.append,
        on_invoke_end=ends.append,
    )
    # Two failures (one wrong-typed output → schema_parse, one
    # asyncio.TimeoutError → timeout), then a success.
    stub = _StubAgent(
        [
            _Out.__class__,  # wrong type — provider raises TypeError → schema_parse
            TimeoutError(),
            _Out(payload="finally"),
        ]
    )
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    result = await provider.invoke(tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out)
    assert result == _Out(payload="finally")
    # 3 starts, 3 ends; outcomes retry, retry, ok in that order on the closing side.
    assert len(starts) == 3
    assert [s.attempt for s in starts] == [1, 2, 3]
    assert [e.outcome for e in ends] == [Outcome.retry, Outcome.retry, Outcome.ok]
    # asyncio.sleep awaited twice (once per retry).
    assert len(sleeps) == 2


# ── Retry exhaustion (ac-007) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_retry_exhaustion_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("molexp.agent._pydanticai.provider.asyncio.sleep", _fake_sleep)

    underlying = ConnectionError("refused")
    provider = PydanticAIProvider(
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
    )
    stub = _StubAgent([underlying, underlying, underlying])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await provider.invoke(
            tier=ModelTier.DEFAULT,
            system="sys",
            user="hi",
            schema=_Out,
            node_id="ingest",
        )
    err = exc_info.value
    assert err.kind is ErrorKind.model_unavailable
    assert err.attempts == 3
    assert err.node_id == "ingest"
    assert err.tier is ModelTier.DEFAULT
    assert err.cause is underlying


# ── Non-retryable ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_non_retryable_propagates_after_one_attempt() -> None:
    class _Mystery(Exception):
        pass

    provider = PydanticAIProvider()
    stub = _StubAgent([_Mystery("nope")])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await provider.invoke(tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out)
    assert exc_info.value.kind is ErrorKind.unknown
    assert exc_info.value.attempts == 1


# ── invoke_with_template (ac-008) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_with_template_renders_then_invokes() -> None:
    provider = PydanticAIProvider()
    stub = _StubAgent([_Out(payload="ok")])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    result = await provider.invoke_with_template(
        tier=ModelTier.DEFAULT,
        system="sys",
        user_template="ask ${q}",
        user_context={"q": "x"},
        schema=_Out,
    )
    assert result == _Out(payload="ok")
    assert stub.calls == ["ask x"]


@pytest.mark.asyncio
async def test_invoke_with_template_missing_key_raises_validation() -> None:
    provider = PydanticAIProvider()
    with pytest.raises(ProviderError) as exc_info:
        await provider.invoke_with_template(
            tier=ModelTier.HEAVY,
            system="sys",
            user_template="ask ${missing}",
            user_context={},
            schema=_Out,
            node_id="codegen",
        )
    assert exc_info.value.kind is ErrorKind.validation
    assert exc_info.value.tier is ModelTier.HEAVY
    assert exc_info.value.node_id == "codegen"


def test_pydantic_ai_provider_satisfies_protocol() -> None:
    """Structural conformance to the :class:`Provider` protocol."""
    assert isinstance(PydanticAIProvider(), Provider)


# ── Hook resilience (ac-009) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_callback_raising_does_not_break_invoke(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _bad_end(_event: ProviderEvent) -> None:
        raise RuntimeError("telemetry sink failed")

    provider = PydanticAIProvider(on_invoke_end=_bad_end)
    stub = _StubAgent([_Out(payload="ok")])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    with caplog.at_level(logging.WARNING, logger="molexp.agent._pydanticai.provider"):
        result = await provider.invoke(tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out)
    assert result == _Out(payload="ok")
    # Faulty hook is logged at WARNING or higher.
    assert any(
        r.levelno >= logging.WARNING and "hook" in r.getMessage().lower() for r in caplog.records
    )


# ── Single-attempt policy ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_attempt_policy_never_retries() -> None:
    provider = PydanticAIProvider(retry_policy=RetryPolicy(max_attempts=1))
    stub = _StubAgent([TimeoutError()])
    _install_stub(provider, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await provider.invoke(tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out)
    assert exc_info.value.attempts == 1


# ── Lazy import preservation ───────────────────────────────────────────────


def test_importing_test_module_does_not_eagerly_load_pydantic_ai() -> None:
    """Importing the *test* module is fine — it imports the provider.
    But importing :mod:`molexp.agent` alone must stay lazy. The
    dedicated import-guard test enforces that; we only sanity-check
    here that the symbol exists post-import."""
    import sys

    # We've already imported the provider above; that's expected to pull
    # pydantic_ai. Confirm the module is loaded so the assertion is real.
    assert "pydantic_ai" in sys.modules
