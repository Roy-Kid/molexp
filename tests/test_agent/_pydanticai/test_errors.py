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


# ── dunder attribute mutability — Python exception machinery ──────────────────


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


def test_classify_pydantic_ai_connection_message_to_model_unavailable() -> None:
    """A transient connection failure surfaces as ``ModelAPIError`` whose class
    name lacks 'HTTP'/'Connection'; the message must still route it to the
    retryable ``model_unavailable`` class (a blip must not kill the plan)."""
    cls = type(
        "ModelAPIError",
        (Exception,),
        {"__module__": "pydantic_ai.exceptions"},
    )
    assert classify(cls("Connection error.")) is ErrorKind.model_unavailable


def test_classify_pydantic_ai_timeout_message_to_timeout() -> None:
    cls = type(
        "ModelAPIError",
        (Exception,),
        {"__module__": "pydantic_ai.exceptions"},
    )
    assert classify(cls("Request timed out")) is ErrorKind.timeout
