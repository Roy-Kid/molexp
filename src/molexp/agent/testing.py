"""Test utilities — FakeModelClient + helpers.

``FakeModelClient`` is the reference implementation for the model
boundary: deterministic, scriptable (pre-seeded responses + tool-call
replay), and the canonical round-trip-correct ``model_io.jsonl``
writer that real plugins crib from.

Lives under ``molexp.agent`` (not ``tests/``) so plugin authors can
import it from outside the test suite.
"""

from __future__ import annotations

from collections import deque
from typing import Any, AsyncIterator, Iterable

from molexp.agent._serialize import to_jsonable
from molexp.agent.model import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from molexp.agent.types import Usage


class ScriptExhausted(RuntimeError):
    """Raised when ``FakeModelClient`` runs out of scripted responses."""


class FakeModelClient:
    """Deterministic, scriptable :class:`ModelClient` implementation.

    Construct with an iterable of scripted ``ModelResponse`` (or
    ``ModelEvent`` lists for streaming). Each call to
    :meth:`complete` / :meth:`stream` pops the next entry. ``calls``
    accumulates the requests received so tests can assert on them.
    """

    def __init__(
        self,
        name: str = "fake",
        responses: Iterable[ModelResponse] = (),
        streams: Iterable[Iterable[ModelEvent]] = (),
    ) -> None:
        self.name = name
        self._responses: deque[ModelResponse] = deque(responses)
        self._streams: deque[list[ModelEvent]] = deque(list(s) for s in streams)
        self.calls: list[ModelRequest] = []
        self.io_log: list[dict[str, Any]] = []

    # ModelClient protocol -------------------------------------------------

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if not self._responses:
            raise ScriptExhausted(
                f"FakeModelClient.complete called {len(self.calls)} times "
                f"but only {len(self.calls) - 1} responses were scripted"
            )
        response = self._responses.popleft()
        self.io_log.append(_io_record(request, response))
        return response

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls.append(request)
        if not self._streams:
            raise ScriptExhausted("FakeModelClient.stream called but no event lists were scripted")
        events = self._streams.popleft()
        return _replay_events(events, request, self.io_log)

    # Scripting helpers ----------------------------------------------------

    def queue_response(self, response: ModelResponse) -> None:
        self._responses.append(response)

    def queue_text(self, text: str, finish_reason: str = "stop") -> None:
        self._responses.append(
            ModelResponse(text=text, finish_reason=finish_reason, usage=Usage(requests=1))
        )

    def queue_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
        finish_reason: str = "tool_calls",
    ) -> None:
        self._responses.append(
            ModelResponse(
                tool_calls=(
                    ModelToolCall(
                        id=call_id or f"call_{len(self.calls)}",
                        name=name,
                        arguments=arguments,
                    ),
                ),
                finish_reason=finish_reason,
                usage=Usage(requests=1),
            )
        )

    def remaining_responses(self) -> int:
        return len(self._responses)


async def _replay_events(
    events: Iterable[ModelEvent],
    request: ModelRequest,
    io_log: list[dict[str, Any]],
) -> AsyncIterator[ModelEvent]:
    materialized = list(events)
    io_log.append(
        {
            "request": _request_to_dict(request),
            "stream_events": [_event_to_dict(e) for e in materialized],
        }
    )
    for event in materialized:
        yield event


def _io_record(request: ModelRequest, response: ModelResponse) -> dict[str, Any]:
    return {
        "request": _request_to_dict(request),
        "response": _response_to_dict(response),
    }


def _request_to_dict(request: ModelRequest) -> dict[str, Any]:
    return to_jsonable(request)


def _response_to_dict(response: ModelResponse) -> dict[str, Any]:
    payload = to_jsonable(response)
    # ``raw`` is opaque to the harness — drop it to keep the JSONL
    # writer round-trip-clean. Plugins that want to preserve raw
    # provider payloads write them out under provider_blobs/ instead.
    payload["raw"] = None
    return payload


def _event_to_dict(event: ModelEvent) -> dict[str, Any]:
    payload = to_jsonable(event)
    payload["raw"] = None
    return payload


__all__ = ["FakeModelClient", "ScriptExhausted"]
