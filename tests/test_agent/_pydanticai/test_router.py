"""Tests for the unified :class:`PydanticAIRouter`.

Drives the router against an in-memory stub of ``pydantic_ai.Agent``
(monkeypatched into the router's ``(tier, schema|None)`` cache so the
real SDK ``Agent`` is never constructed). Covers the structured path's
retry / event hardening (ported from the deleted
``test_provider_hardening``) plus the new text path.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from molexp.agent._pydanticai.errors import ErrorKind, ProviderError
from molexp.agent._pydanticai.events import Outcome, ProviderEvent
from molexp.agent._pydanticai.retry import RetryPolicy, should_retry
from molexp.agent._pydanticai.router import PydanticAIRouter
from molexp.agent.router import ModelTier, Router, RouterTextResult


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
    return {
        ModelTier.CHEAP: model,
        ModelTier.DEFAULT: model,
        ModelTier.HEAVY: model,
    }


def _install_structured_stub(
    router: PydanticAIRouter,
    tier: ModelTier,
    schema: type[BaseModel],
    stub: _StubAgent,
) -> None:
    """Bypass ``_structured_agent`` by pre-populating the cache."""
    router._agents[(tier, schema)] = stub  # type: ignore[assignment]


def _install_text_stub(
    router: PydanticAIRouter,
    tier: ModelTier,
    stub: _StubAgent,
) -> None:
    """Bypass ``_text_agent`` by pre-populating the cache."""
    router._agents[(tier, None)] = stub  # type: ignore[assignment]


# ── Construction validation ────────────────────────────────────────────────


def test_router_construction_requires_all_tiers() -> None:
    with pytest.raises(ValueError, match="must cover every ModelTier"):
        PydanticAIRouter(models={ModelTier.DEFAULT: "x"})


def test_router_construction_rejects_none_value() -> None:
    with pytest.raises(ValueError, match="may not be None"):
        PydanticAIRouter(
            models={
                ModelTier.CHEAP: "x",
                ModelTier.DEFAULT: None,  # type: ignore[dict-item]
                ModelTier.HEAVY: "y",
            }
        )


# ── complete_structured: happy path ────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_structured_happy_path_one_attempt() -> None:
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


# ── complete_structured: retry on schema parse / timeout ──────────────────


@pytest.mark.asyncio
async def test_complete_structured_retries_two_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
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
    stub = _StubAgent(
        [
            TimeoutError(),
            TimeoutError(),
            _Out(payload="finally"),
        ]
    )
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    result = await router.complete_structured(
        tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
    )
    assert result == _Out(payload="finally")
    assert len(starts) == 3
    assert [s.attempt for s in starts] == [1, 2, 3]
    assert [e.outcome for e in ends] == [Outcome.retry, Outcome.retry, Outcome.ok]
    assert len(sleeps) == 2


# ── complete_structured: retry exhaustion ─────────────────────────────────


@pytest.mark.asyncio
async def test_complete_structured_retry_exhaustion_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
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


# ── complete_structured: non-retryable propagates after one attempt ───────


@pytest.mark.asyncio
async def test_complete_structured_non_retryable_propagates_after_one_attempt() -> None:
    class _Mystery(Exception):
        pass

    router = PydanticAIRouter(models=_models_all("x"))
    stub = _StubAgent([_Mystery("nope")])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
    assert exc_info.value.kind is ErrorKind.unknown
    assert exc_info.value.attempts == 1


# ── Hook resilience ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_callback_raising_does_not_break_invoke() -> None:
    def _bad_end(_event: ProviderEvent) -> None:
        raise RuntimeError("telemetry sink failed")

    router = PydanticAIRouter(models=_models_all("x"), on_invoke_end=_bad_end)
    stub = _StubAgent([_Out(payload="ok")])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    # Contract: a faulty telemetry sink must not poison the LLM call
    # path. The router catches and logs the hook exception; the call
    # must still return the schema-typed result.
    result = await router.complete_structured(
        tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
    )
    assert result == _Out(payload="ok")


# ── Single-attempt policy ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_attempt_policy_never_retries() -> None:
    router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    stub = _StubAgent([TimeoutError()])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
    assert exc_info.value.attempts == 1


# ── Cache hits across calls ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structured_agent_cached_per_tier_schema_pair() -> None:
    """Two calls at the same (tier, schema) must reuse one Agent."""
    router = PydanticAIRouter(models=_models_all("x"))
    stub = _StubAgent([_Out(payload="a"), _Out(payload="b")])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    out1 = await router.complete_structured(
        tier=ModelTier.DEFAULT, system="sys", user="1", schema=_Out
    )
    out2 = await router.complete_structured(
        tier=ModelTier.DEFAULT, system="sys", user="2", schema=_Out
    )
    assert out1.payload == "a"
    assert out2.payload == "b"
    assert stub.calls == ["1", "2"]


# ── complete_text: text path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_text_returns_router_text_result() -> None:
    router = PydanticAIRouter(models=_models_all("x"))
    stub = _StubAgent(["hello back"])
    _install_text_stub(router, ModelTier.DEFAULT, stub)

    result = await router.complete_text(prompt="hi")
    assert isinstance(result, RouterTextResult)
    assert result.text == "hello back"
    assert stub.calls == ["hi"]


@pytest.mark.asyncio
async def test_complete_text_uses_requested_tier() -> None:
    """Each tier has its own cache slot; routing the call hits the right stub."""
    router = PydanticAIRouter(models=_models_all("x"))
    cheap_stub = _StubAgent(["cheap reply"])
    heavy_stub = _StubAgent(["heavy reply"])
    _install_text_stub(router, ModelTier.CHEAP, cheap_stub)
    _install_text_stub(router, ModelTier.HEAVY, heavy_stub)

    cheap_out = await router.complete_text(prompt="hi", tier=ModelTier.CHEAP)
    heavy_out = await router.complete_text(prompt="hi", tier=ModelTier.HEAVY)
    assert cheap_out.text == "cheap reply"
    assert heavy_out.text == "heavy reply"
    assert cheap_stub.calls == ["hi"]
    assert heavy_stub.calls == ["hi"]


# ── Transport-retry hardening: complete_text resilience (ac-001) ───────────


@pytest.mark.asyncio
async def test_complete_text_recovers_from_first_attempt_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-001 — RED-first driver for the one new behavior.

    The text path must recover from a first-attempt transport failure
    exactly as ``complete_structured`` already does. ``TimeoutError``
    is classified by :func:`classify` as :attr:`ErrorKind.timeout`,
    which is in the default :class:`RetryPolicy.retry_on`, so attempt
    #1 retries and attempt #2 succeeds.

    Fails today: ``complete_text`` issues a single ``agent.run`` and
    re-raises, so only one call is recorded and the exception
    propagates instead of recovering.
    """

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
    assert isinstance(result, RouterTextResult)
    assert result.text == "recovered"
    assert stub.calls == ["hi", "hi"]


@pytest.mark.asyncio
async def test_complete_text_recovers_from_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-001 — same recovery via the ``model_unavailable`` kind.

    ``ConnectionError`` subclasses ``OSError`` which :func:`classify`
    maps to :attr:`ErrorKind.model_unavailable`, also retryable by
    default. Confirms the text path covers both default-retryable
    transport kinds.
    """

    async def _fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
    )
    stub = _StubAgent([ConnectionError("refused"), "recovered"])
    _install_text_stub(router, ModelTier.DEFAULT, stub)

    result = await router.complete_text(prompt="hi")
    assert result.text == "recovered"
    assert stub.calls == ["hi", "hi"]


# ── Transport-retry hardening: shared helper / parity (ac-002) ─────────────


@pytest.mark.asyncio
async def test_text_and_structured_share_identical_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-002 — behavioral parity: both paths retry transport errors
    with identical attempt counts.

    Drives the same transport-failure-then-success script through
    ``complete_text`` and ``complete_structured`` and asserts the stub
    recorded the same number of ``run()`` calls in each, proving both
    route their model invocation through one shared transport-retry
    mechanism rather than divergent hand-rolled loops.
    """

    async def _fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    text_router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
    )
    text_stub = _StubAgent([TimeoutError(), TimeoutError(), "ok"])
    _install_text_stub(text_router, ModelTier.DEFAULT, text_stub)
    text_result = await text_router.complete_text(prompt="hi")
    assert text_result.text == "ok"

    struct_router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
    )
    struct_stub = _StubAgent([TimeoutError(), TimeoutError(), _Out(payload="ok")])
    _install_structured_stub(struct_router, ModelTier.DEFAULT, _Out, struct_stub)
    struct_result = await struct_router.complete_structured(
        tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
    )
    assert struct_result == _Out(payload="ok")

    # Identical attempt counts prove a shared transport-retry mechanism.
    assert len(text_stub.calls) == len(struct_stub.calls) == 3


def test_router_exposes_single_transport_retry_helper() -> None:
    """ac-002 — structural lock: one private transport-retry helper
    exists on the class and ``complete_text`` no longer hand-rolls its
    own retry loop.

    Lenient: asserts the helper symbol exists and that
    ``complete_text``'s source references it rather than embedding a
    ``while True`` retry loop of its own.
    """
    import inspect

    assert hasattr(PydanticAIRouter, "_run_with_transport_retry")

    text_src = inspect.getsource(PydanticAIRouter.complete_text)
    assert "_run_with_transport_retry" in text_src
    assert "while True" not in text_src


# ── Transport-retry hardening: parse/validation not multiplied (ac-004) ────


@pytest.mark.asyncio
async def test_schema_parse_not_multiplied_by_router_loop() -> None:
    """ac-004 — prod-incident regression lock.

    A wrong-typed structured output triggers the router's own
    ``TypeError`` schema-mismatch raise, which :func:`classify` maps to
    :attr:`ErrorKind.schema_parse`. Since ``schema_parse`` is absent
    from the default ``RetryPolicy.retry_on``, the router must NOT
    retry: exactly one ``run()`` call, then a terminal
    ``ProviderError(kind=schema_parse, attempts=1)``.
    """
    router = PydanticAIRouter(models=_models_all("x"))
    # Returning a non-``_Out`` value makes the router's isinstance check
    # raise TypeError -> classify -> schema_parse.
    stub = _StubAgent(["a bare string, not an _Out instance"])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
    assert exc_info.value.kind is ErrorKind.schema_parse
    assert exc_info.value.attempts == 1
    assert stub.calls == ["hi"]


def test_schema_parse_and_validation_absent_from_default_retry_on() -> None:
    """ac-004 — confirm the no-multiply kinds stay out of the policy."""
    policy = RetryPolicy()
    assert ErrorKind.schema_parse not in policy.retry_on
    assert ErrorKind.validation not in policy.retry_on


# ── Transport-retry hardening: terminal exhaustion (ac-005) ────────────────


@pytest.mark.asyncio
async def test_complete_text_terminal_transport_failure_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-005 — text path: exhausting the budget with repeated
    transport errors raises ``ProviderError`` whose ``kind`` is the
    transport kind and whose ``attempts`` equals ``max_attempts``.
    """

    async def _fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    router = PydanticAIRouter(models=_models_all("x"), retry_policy=policy)
    stub = _StubAgent([TimeoutError(), TimeoutError(), TimeoutError()])
    _install_text_stub(router, ModelTier.DEFAULT, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_text(prompt="hi")
    assert exc_info.value.kind is ErrorKind.timeout
    assert exc_info.value.attempts == policy.max_attempts == 3
    assert stub.calls == ["hi", "hi", "hi"]


@pytest.mark.asyncio
async def test_complete_structured_terminal_transport_failure_preserves_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-005 — structured path: terminal transport failure preserves
    ``attempts == max_attempts`` (lock on already-correct behavior)."""

    async def _fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    router = PydanticAIRouter(models=_models_all("x"), retry_policy=policy)
    stub = _StubAgent([TimeoutError(), TimeoutError(), TimeoutError()])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
    assert exc_info.value.kind is ErrorKind.timeout
    assert exc_info.value.attempts == 3


# ── Transport-retry hardening: validation/unknown no retry (ac-006) ────────


@pytest.mark.asyncio
async def test_unknown_error_propagates_without_retry_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-006 — an unknown-classified error propagates as
    ``ProviderError`` with ``attempts == 1`` and a single ``run()``
    call (no retry sleep) on the text path.

    NOTE on the discrepancy (see report): no exception in
    :func:`classify` ever maps to :attr:`ErrorKind.validation` — it is
    a defined-but-never-produced member. The acceptance text pairs
    "validation- and unknown-classified". This test exercises the
    ``unknown`` path behaviorally; the policy-level fact that
    ``validation`` is non-retryable is locked by
    :func:`test_validation_kind_is_not_retryable`.
    """
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    class _Mystery(Exception):
        pass

    router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    stub = _StubAgent([_Mystery("nope")])
    _install_text_stub(router, ModelTier.DEFAULT, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_text(prompt="hi")
    assert exc_info.value.kind is ErrorKind.unknown
    assert exc_info.value.attempts == 1
    assert stub.calls == ["hi"]
    assert sleeps == []


@pytest.mark.asyncio
async def test_unknown_error_propagates_without_retry_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac-006 — structured path: unknown-classified error propagates
    with ``attempts == 1`` and no retry sleep (lock)."""
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("molexp.agent._pydanticai.router.asyncio.sleep", _fake_sleep)

    class _Mystery(Exception):
        pass

    router = PydanticAIRouter(
        models=_models_all("x"),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    stub = _StubAgent([_Mystery("nope")])
    _install_structured_stub(router, ModelTier.DEFAULT, _Out, stub)

    with pytest.raises(ProviderError) as exc_info:
        await router.complete_structured(
            tier=ModelTier.DEFAULT, system="sys", user="hi", schema=_Out
        )
    assert exc_info.value.kind is ErrorKind.unknown
    assert exc_info.value.attempts == 1
    assert stub.calls == ["hi"]
    assert sleeps == []


def test_validation_kind_is_not_retryable() -> None:
    """ac-006 — policy-level lock: ``validation`` is not in the default
    ``retry_on``, so should it ever be produced it would not be
    retried."""
    policy = RetryPolicy()
    assert ErrorKind.validation not in policy.retry_on
    assert should_retry(ErrorKind.validation, policy, attempt=1) is False
    assert should_retry(ErrorKind.unknown, policy, attempt=1) is False


# ── Protocol conformance ──────────────────────────────────────────────────


def test_pydantic_ai_router_satisfies_protocol() -> None:
    """Structural conformance to the :class:`Router` protocol."""
    assert isinstance(PydanticAIRouter(models=_models_all("x")), Router)


# ── Lazy import preservation ───────────────────────────────────────────────


def test_importing_test_module_does_not_eagerly_load_pydantic_ai() -> None:
    """Importing the *test* module is fine — it imports the router. But
    importing :mod:`molexp.agent` alone must stay lazy. The dedicated
    import-guard test enforces that; we only sanity-check here that the
    SDK is in fact loaded once we touched the router class."""
    import sys

    assert "pydantic_ai" in sys.modules
