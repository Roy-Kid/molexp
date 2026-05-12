"""JSON codec for pydantic-ai ``ModelMessage`` — the on-disk session history.

The agent layer keeps a ``model_messages`` field on
:class:`~molexp.agent.session.AgentSession` so that the LLM-native
conversation context survives across turns (and across process
restarts, when paired with :class:`~molexp.agent.sessions.SessionCatalog`).

The shape of those messages — :class:`pydantic_ai.messages.ModelMessage` —
is owned by pydantic-ai. Per the agent layer's import-boundary
firewall, only files under ``_pydanticai/`` may import that SDK; this
module is the sole serialization site.

We use pydantic-ai's official type adapter
(:class:`pydantic_ai.ModelMessagesTypeAdapter`) so any round-trip
remains stable across pydantic-ai versions.

Design notes
============

* The codec validates an opaque ``Iterable[Any]`` on the write side and
  returns ``tuple[Any, ...]`` on the read side. Callers (the agent
  layer's :class:`SessionCatalog`) treat the elements as opaque values
  — only this module reaches into the pydantic-ai type.
* ``ModelMessagesTypeAdapter`` is a ``TypeAdapter[list[ModelMessage]]``;
  ``validate_json`` returns ``list[ModelMessage]`` which we coerce to a
  tuple so the caller's persisted state is immutable by convention.
* Empty / missing on-disk files are not this module's concern —
  :class:`SessionCatalog` checks for ``Path.exists()`` before calling
  :func:`load_model_messages`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter

__all__ = ["dump_model_messages", "load_model_messages"]


def dump_model_messages(messages: Iterable[Any]) -> bytes:
    """Serialize ``messages`` to canonical pydantic-ai JSON bytes.

    Args:
        messages: An iterable of
            :class:`pydantic_ai.messages.ModelMessage` instances. The
            caller passes them opaquely as ``Any``; this module is the
            only place that knows the concrete type.

    Returns:
        UTF-8-encoded JSON bytes safe to write to disk.
    """
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def load_model_messages(data: bytes) -> tuple[Any, ...]:
    """Parse pydantic-ai JSON bytes back into a ``ModelMessage`` tuple.

    Args:
        data: Bytes previously produced by :func:`dump_model_messages`.

    Returns:
        Tuple of :class:`pydantic_ai.messages.ModelMessage` instances
        (typed ``Any`` at the boundary so the agent layer stays free of
        ``pydantic_ai`` imports outside this subpackage).

    Raises:
        pydantic.ValidationError: If ``data`` does not validate against
            pydantic-ai's current ``ModelMessage`` schema. Callers that
            want graceful fallback (e.g. on a version skew) should
            catch this and treat the session as fresh.
    """
    parsed = ModelMessagesTypeAdapter.validate_json(data)
    return tuple(parsed)
