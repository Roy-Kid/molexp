"""Tests for :mod:`molexp.agent._pydanticai.errors`.

Two behaviors are owned here:

- :class:`ProviderError` is immutable on its documented fields but keeps its
  dunder attributes writable so Python's exception machinery can propagate it.
- :func:`classify` maps each exception family — real ``isinstance`` cases and
  the ``pydantic_ai.*`` SDK exceptions detected structurally by name/message —
  onto the right :class:`ErrorKind`.
"""

from __future__ import annotations

import pydantic
import pytest

from molexp.agent._pydanticai.errors import ErrorKind, ProviderError, classify
from molexp.agent.router import ModelTier


class TestProviderError:
    def test_documented_fields_are_immutable_after_construction(self) -> None:
        err = ProviderError(ErrorKind.unknown, node_id="", tier=ModelTier.DEFAULT)
        with pytest.raises(AttributeError):
            err.kind = ErrorKind.timeout  # type: ignore[misc]
        with pytest.raises(AttributeError):
            err.attempts = 99  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_dunder_attributes_stay_mutable_for_exception_propagation(self) -> None:
        """Regression: every mode runs inside ``harness.stage()`` (an
        ``@asynccontextmanager``); contextlib's ``__aexit__`` assigns
        ``exc.__traceback__``. If ``__setattr__`` blocked dunders, that write
        raised a masking ``AttributeError`` and crashed the run instead of
        surfacing the original ``ProviderError``.
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


class TestClassify:
    def test_pydantic_validation_error_is_schema_parse(self) -> None:
        class _M(pydantic.BaseModel):
            n: int

        try:
            _M(n="not-an-int")  # type: ignore[arg-type]
        except pydantic.ValidationError as exc:
            assert classify(exc) is ErrorKind.schema_parse
        else:  # pragma: no cover — pydantic must reject
            pytest.fail("expected ValidationError")

    def test_timeout_error_is_timeout(self) -> None:
        assert classify(TimeoutError()) is ErrorKind.timeout

    def test_type_error_is_schema_parse(self) -> None:
        """The wrapper's own isinstance schema-mismatch raise surfaces as
        ``TypeError`` and must classify as schema_parse."""
        assert classify(TypeError("expected Foo; received Bar")) is ErrorKind.schema_parse

    def test_os_error_is_model_unavailable(self) -> None:
        assert classify(ConnectionError("refused")) is ErrorKind.model_unavailable

    def test_unrecognized_exception_is_unknown(self) -> None:
        """A bare ``ValueError`` must fall through to ``unknown`` — the
        ``pydantic.ValidationError`` check is deliberately specific, not a
        catch-all for its ``ValueError`` base."""
        assert classify(ValueError("bare")) is ErrorKind.unknown

    @pytest.mark.parametrize(
        ("class_name", "message", "expected"),
        [
            # SDK exceptions are detected by module path (no eager SDK import).
            ("UnexpectedModelBehavior", "oops", ErrorKind.schema_parse),  # by name
            ("ModelHTTPError", "boom", ErrorKind.model_unavailable),  # by name
            # Generic ``ModelAPIError`` (name lacks HTTP/Connection) → by message,
            # so a transient blip retries instead of aborting the pipeline.
            ("ModelAPIError", "Connection error.", ErrorKind.model_unavailable),
            ("ModelAPIError", "Request timed out", ErrorKind.timeout),
        ],
    )
    def test_pydantic_ai_sdk_exception_classified_by_name_then_message(
        self, class_name: str, message: str, expected: ErrorKind
    ) -> None:
        cls = type(class_name, (Exception,), {"__module__": "pydantic_ai.exceptions"})
        assert classify(cls(message)) is expected
