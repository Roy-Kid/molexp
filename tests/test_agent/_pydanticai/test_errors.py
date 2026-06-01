"""Tests for :mod:`molexp.agent._pydanticai.errors`.

Coverage focus (acceptance criteria ac-001 + ac-002):

- :class:`ErrorKind` enum membership matches the documented set.
- :class:`ProviderError` carries the documented frozen attributes
  via ``__slots__``; mutation is rejected.
- :func:`classify` maps each known exception family to the right
  :class:`ErrorKind`; unrecognized exceptions fall back to
  ``ErrorKind.unknown``.
"""

from __future__ import annotations

import pydantic
import pytest

from molexp.agent._pydanticai.errors import (
    ErrorKind,
    ProviderError,
    classify,
)
from molexp.agent.router import ModelTier

# ── ErrorKind enum ─────────────────────────────────────────────────────────


def test_error_kind_members_match_documented_set() -> None:
    expected = {
        "model_unavailable",
        "schema_parse",
        "validation",
        "timeout",
        "unknown",
    }
    actual = {member.value for member in ErrorKind}
    assert actual == expected


# ── ProviderError ──────────────────────────────────────────────────────────


def test_provider_error_carries_documented_attributes() -> None:
    cause = RuntimeError("inner")
    err = ProviderError(
        ErrorKind.timeout,
        node_id="ingest",
        tier=ModelTier.CHEAP,
        cause=cause,
        attempts=2,
    )
    assert err.kind is ErrorKind.timeout
    assert err.node_id == "ingest"
    assert err.tier is ModelTier.CHEAP
    assert err.cause is cause
    assert err.attempts == 2


def test_provider_error_default_attempts_is_one() -> None:
    err = ProviderError(
        ErrorKind.unknown,
        node_id="x",
        tier=ModelTier.DEFAULT,
    )
    assert err.attempts == 1
    assert err.cause is None


def test_provider_error_documented_fields_are_immutable() -> None:
    err = ProviderError(
        ErrorKind.unknown,
        node_id="",
        tier=ModelTier.DEFAULT,
    )
    with pytest.raises(AttributeError):
        err.kind = ErrorKind.timeout  # type: ignore[misc]
    with pytest.raises(AttributeError):
        err.attempts = 99  # type: ignore[misc]


def test_provider_error_rejects_undeclared_post_init_attributes() -> None:
    err = ProviderError(
        ErrorKind.unknown,
        node_id="",
        tier=ModelTier.DEFAULT,
    )
    with pytest.raises(AttributeError):
        err.extra_field = "no"  # type: ignore[attr-defined]


def test_provider_error_message_default_includes_kind_and_attempts() -> None:
    err = ProviderError(
        ErrorKind.schema_parse,
        node_id="codegen",
        tier=ModelTier.HEAVY,
        attempts=3,
    )
    text = str(err)
    assert "schema_parse" in text
    assert "codegen" in text
    assert "heavy" in text
    assert "3" in text


def test_provider_error_message_explicit_overrides_default() -> None:
    err = ProviderError(
        ErrorKind.unknown,
        node_id="x",
        tier=ModelTier.DEFAULT,
        message="custom",
    )
    assert str(err) == "custom"


# ── dunder attribute mutability — Python exception machinery ──────────────────


def test_provider_error_traceback_can_be_set_post_init() -> None:
    """Python's exception propagation sets ``__traceback__``; the immutability
    guard must not block dunder attribute writes."""
    err = ProviderError(ErrorKind.unknown, node_id="x", tier=ModelTier.DEFAULT)
    try:
        raise RuntimeError("synthesize a traceback")
    except RuntimeError as inner:
        err.__traceback__ = inner.__traceback__
    assert err.__traceback__ is not None


def test_provider_error_raise_from_sets_cause() -> None:
    """``raise X from Y`` sets ``X.__cause__``; must not crash on the frozen guard."""
    inner = RuntimeError("inner")
    err = ProviderError(ErrorKind.unknown, node_id="x", tier=ModelTier.DEFAULT)
    with pytest.raises(ProviderError) as exc_info:
        raise err from inner
    assert exc_info.value.__cause__ is inner


@pytest.mark.asyncio
async def test_provider_error_propagates_through_async_context_manager() -> None:
    """A ProviderError raised inside ``@asynccontextmanager`` must surface as
    itself, not be masked by a secondary ``AttributeError`` on
    ``__traceback__`` assignment in contextlib's ``__aexit__``.

    Regression: every molexp mode runs inside ``harness.stage()`` (an
    ``@asynccontextmanager``); without this guarantee, any ``ProviderError``
    inside a stage crashed the whole run instead of propagating.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def stage():
        yield

    err = ProviderError(ErrorKind.unknown, node_id="x", tier=ModelTier.DEFAULT)
    with pytest.raises(ProviderError) as exc_info:
        async with stage():
            raise err
    assert exc_info.value is err


# ── classify ───────────────────────────────────────────────────────────────


def test_classify_pydantic_validation_error_to_schema_parse() -> None:
    class _M(pydantic.BaseModel):
        n: int

    try:
        _M(n="not-an-int")  # type: ignore[arg-type]
    except pydantic.ValidationError as exc:
        assert classify(exc) is ErrorKind.schema_parse
    else:  # pragma: no cover — pydantic must reject
        pytest.fail("expected ValidationError")


def test_classify_asyncio_timeout_error_to_timeout() -> None:
    assert classify(TimeoutError()) is ErrorKind.timeout


def test_classify_builtin_timeout_error_to_timeout() -> None:
    # In Python 3.11+ asyncio.TimeoutError is aliased to TimeoutError;
    # classify recognizes either.
    assert classify(TimeoutError()) is ErrorKind.timeout


def test_classify_type_error_schema_mismatch_to_schema_parse() -> None:
    """The provider raises ``TypeError`` on isinstance schema mismatch."""
    err = TypeError("Provider expected Foo; received Bar")
    assert classify(err) is ErrorKind.schema_parse


def test_classify_os_error_to_model_unavailable() -> None:
    assert classify(ConnectionError("refused")) is ErrorKind.model_unavailable
    assert classify(OSError(111, "socket")) is ErrorKind.model_unavailable


def test_classify_unknown_exception_to_unknown() -> None:
    class CustomError(Exception):
        pass

    assert classify(CustomError("bizarre")) is ErrorKind.unknown
    assert classify(ValueError("bare")) is ErrorKind.unknown


def test_classify_pydantic_ai_validation_like_to_schema_parse() -> None:
    """Names matching Validation/Schema/UnexpectedModel under
    pydantic_ai.* classify as schema_parse without importing the SDK."""
    cls = type(
        "UnexpectedModelBehavior",
        (Exception,),
        {"__module__": "pydantic_ai.exceptions"},
    )
    assert classify(cls("oops")) is ErrorKind.schema_parse


def test_classify_pydantic_ai_http_like_to_model_unavailable() -> None:
    cls = type(
        "ModelHTTPError",
        (Exception,),
        {"__module__": "pydantic_ai.exceptions"},
    )
    assert classify(cls("boom")) is ErrorKind.model_unavailable
