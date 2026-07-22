"""JSON codec for pydantic-ai ``ModelMessage`` — the on-disk session history.

The agent layer keeps a ``model_messages`` field on
:class:`~molexp.agent.session.AgentSession` so that the LLM-native
conversation context survives across turns (and across process
restarts, when paired with :class:`~molexp.agent.folders.AgentSession`
— the on-disk ``Folder`` subclass that persists each session's
``messages.jsonl``).

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
  layer's :class:`~molexp.agent.folders.AgentSession`) treat the
  elements as opaque values — only this module reaches into the
  pydantic-ai type.
* ``ModelMessagesTypeAdapter`` is a ``TypeAdapter[list[ModelMessage]]``;
  ``validate_json`` returns ``list[ModelMessage]`` which we coerce to a
  tuple so the caller's persisted state is immutable by convention.
* Empty / missing on-disk files are not this module's concern —
  :class:`~molexp.agent.folders.AgentSession` checks for
  ``Path.exists()`` before calling :func:`load_model_messages`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelMessagesTypeAdapter

if TYPE_CHECKING:
    from molexp.agent.types import Message

__all__ = [
    "dump_model_messages",
    "load_model_messages",
    "model_messages_from_messages",
]

# Fixed reconstruction timestamp. pydantic-ai stamps each freshly-built
# ``UserPromptPart`` / ``SystemPromptPart`` / ``ModelResponse`` with
# ``now()`` by default, which would make two rebuilds of the *same*
# conversation compare unequal. Pinning a constant makes
# :func:`model_messages_from_messages` deterministic — a required property,
# since the reseed bridge is compared against itself across turns.
_RESEED_TIMESTAMP = datetime(1970, 1, 1, tzinfo=UTC)


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


def model_messages_from_messages(messages: Iterable[Message]) -> tuple[Any, ...]:
    """Rebuild pydantic-ai ``ModelMessage``\\ s from molexp :class:`Message`\\ s.

    The *reseed bridge*: when the lossless ``model_messages`` blob is
    discarded (a branch/resume moved the session tip, or the blob was
    deleted), :class:`~molexp.agent.loops.interactive.InteractiveLoop`
    rebuilds LLM history from the canonical entry tree
    (``session.build_context()``) through this function. The rebuild is
    *semantic* — it carries role + text only, so tool-call detail the lossless
    blob preserved is intentionally absent.

    Role mapping:

    * ``user`` → a ``ModelRequest`` with a single ``UserPromptPart``.
    * ``system`` → a ``ModelRequest`` with a single ``SystemPromptPart``.
    * everything else (``assistant`` / ``tool``) → a ``ModelResponse`` with a
      single ``TextPart``.

    Args:
        messages: molexp conversation turns (``role`` + ``content``).

    Returns:
        A tuple of pydantic-ai ``ModelMessage`` instances, typed ``Any`` at
        the boundary so callers stay free of ``pydantic_ai`` imports. Two
        calls with equal input produce equal output (the timestamp is pinned).
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    rebuilt: list[Any] = []
    for message in messages:
        if message.role == "system":
            rebuilt.append(
                ModelRequest(
                    parts=[SystemPromptPart(content=message.content, timestamp=_RESEED_TIMESTAMP)]
                )
            )
        elif message.role == "user":
            rebuilt.append(
                ModelRequest(
                    parts=[UserPromptPart(content=message.content, timestamp=_RESEED_TIMESTAMP)]
                )
            )
        else:
            rebuilt.append(
                ModelResponse(
                    parts=[TextPart(content=message.content)], timestamp=_RESEED_TIMESTAMP
                )
            )
    return tuple(rebuilt)
