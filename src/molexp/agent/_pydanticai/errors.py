"""Normalized provider errors — single exception type the wrapper raises.

Callers of :class:`PydanticAIRouter` see exactly one error class
(:class:`ProviderError`) regardless of which underlying SDK / network
layer surfaced the failure. The :func:`classify` helper maps the
known exception families (``pydantic.ValidationError``,
``asyncio.TimeoutError``, ``OSError`` / connection errors, the
provider's own ``TypeError`` schema-mismatch raise, and
``pydantic_ai`` SDK exceptions detected by module name to avoid
eager-loading the SDK) onto a small :class:`ErrorKind` enum that the
retry policy can pattern-match against.

This module **does not import** ``pydantic_ai``. ``classify`` checks
for SDK exceptions structurally (by module path / class name) so the
import-guard test ``test_importing_molexp_agent_does_not_load_pydantic_ai``
keeps passing.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

import pydantic

from molexp.agent.router import ModelTier

__all__ = [
    "ErrorKind",
    "ProviderError",
    "classify",
]


class ErrorKind(StrEnum):
    """Normalized failure classes the retry policy can target."""

    model_unavailable = "model_unavailable"
    """Network / HTTP / connection failure reaching the model."""

    schema_parse = "schema_parse"
    """Model returned malformed JSON or a wrong-typed structured output."""

    validation = "validation"
    """Caller-side input was malformed (e.g. template rendering failed)."""

    timeout = "timeout"
    """Operation exceeded its time budget."""

    unknown = "unknown"
    """Anything :func:`classify` did not recognize. Not retryable by default."""


class ProviderError(Exception):
    """Single normalized exception every provider failure path raises.

    Data attributes are immutable post-construction: ``__init__`` is
    the only writer of the documented fields, and any subsequent
    ``setattr`` of a non-dunder attribute raises :class:`AttributeError`.
    Dunder attributes (notably ``__traceback__`` / ``__cause__`` /
    ``__context__`` / ``__suppress_context__``) remain mutable so
    Python's exception-propagation machinery can populate them during
    raise / re-raise — blocking them would crash any context manager
    that re-raises with a secondary ``AttributeError``. Pydantic is
    intentionally not used here because it does not subclass
    :class:`Exception` cleanly.

    Attributes:
        kind: Classified failure class.
        node_id: Caller-supplied identifier for the workflow node that
            triggered the call (empty string when unset).
        tier: Model tier that was being invoked.
        cause: Underlying exception, if any. Distinct from
            ``__cause__`` (which Python sets via ``raise ... from``);
            both point at the same object once the wrapper raises.
        attempts: Number of attempts made before this error fired
            (``1`` for non-retryable failures, up to ``max_attempts``
            for retry exhaustion).
    """

    _FROZEN_FIELDS = frozenset({"kind", "node_id", "tier", "cause", "attempts"})

    def __init__(
        self,
        kind: ErrorKind,
        *,
        node_id: str,
        tier: ModelTier,
        cause: BaseException | None = None,
        attempts: int = 1,
        message: str | None = None,
    ) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ during init.
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "cause", cause)
        object.__setattr__(self, "attempts", attempts)
        super().__init__(
            message
            or (
                f"provider error ({kind.value}) at "
                f"node={node_id!r} tier={tier.value} after {attempts} attempt(s)"
            )
        )
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        # Always allow dunder attributes — Python's exception machinery
        # sets `__traceback__` / `__cause__` / `__context__` /
        # `__suppress_context__` during propagation (e.g. contextlib's
        # `_GeneratorContextManager.__aexit__` does `exc.__traceback__ =
        # traceback`), and blocking them would crash every
        # `async with harness.stage(): ... raise ProviderError(...)`
        # path with a masking AttributeError.
        if name.startswith("__") and name.endswith("__"):
            object.__setattr__(self, name, value)
            return
        # Once __init__ has completed, reject all other attribute writes.
        # During init, _initialized has not been set yet so the guard passes.
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"ProviderError is immutable after construction; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)


def classify(exc: BaseException) -> ErrorKind:
    """Map ``exc`` to one of the :class:`ErrorKind` members.

    Order matters — :class:`pydantic.ValidationError` is checked
    before generic :class:`ValueError` because the former is a
    subclass and we want the more specific classification. SDK
    exceptions (``pydantic_ai.*``) are detected by module path
    instead of by ``isinstance`` to avoid eager-loading the SDK.
    """
    if isinstance(exc, pydantic.ValidationError):
        return ErrorKind.schema_parse
    if isinstance(exc, asyncio.TimeoutError):
        return ErrorKind.timeout
    if isinstance(exc, TimeoutError):  # stdlib alias since 3.11
        return ErrorKind.timeout
    if isinstance(exc, TypeError):
        # The wrapper's own isinstance schema-mismatch raise.
        return ErrorKind.schema_parse
    if isinstance(exc, OSError):
        # ConnectionError, FileNotFoundError, etc. all subclass OSError.
        return ErrorKind.model_unavailable

    cls = type(exc)
    module = getattr(cls, "__module__", "") or ""
    name = cls.__name__
    if module.startswith("pydantic_ai"):
        if "Validation" in name or "Schema" in name or "UnexpectedModel" in name:
            return ErrorKind.schema_parse
        if "HTTP" in name or "Connection" in name:
            return ErrorKind.model_unavailable
    return ErrorKind.unknown
