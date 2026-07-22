"""Entry tree is canonical; the model-messages blob is a reconcilable cache.

Behavior-based RED tests for ac-005 of ``plan-emergent-02-loop-refactor``.
The session entry tree (``entries.jsonl``) is the authoritative history;
the lossless pydantic-ai ``model_messages`` blob beside it
(``messages.jsonl``) is a *cache* reconciled against the active leaf:

* (a) linear continuation, matching leaf stamp -> feed the lossless blob
  (it carries tool-call detail the semantic ``build_context()`` drops).
* (b) branch to a sibling -> the stale blob is discarded and history is
  reseeded from the entry tree via ``model_messages_from_messages``; the
  other branch's model history is never inherited.
* (c) ``messages.jsonl`` deleted -> the loop still reconstructs context
  from the entry tree (no crash), history rebuilt from ``build_context()``.

The loop is driven with ``hooks=None`` (today's single-pass behavior) so
each ``loop.run`` is exactly one ReAct pass; two/three runs give the
successive passes these tests inspect. A scripted fake Router lets each
pass plant a controlled ``FinalChunk.model_messages_json`` blob. A real
``JsonlSessionStorage`` in ``tmp_path`` backs the entry tree.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from molexp.agent._pydanticai.messages_codec import (
    dump_model_messages,
    load_model_messages,
    model_messages_from_messages,
)
from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AgentEvent,
    AsyncIteratorEventSink,
    LoopCompletedEvent,
)
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.folders import MESSAGES_FILENAME
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    McpToolSpec,
    ModelTier,
    RouterTextResult,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import JsonlSessionStorage
from molexp.agent.types import Message, UsageBreakdown

pytestmark = pytest.mark.asyncio

_PLANTED_TOOL = "planted_tool"


def _blob_with_tool(user_text: str, final_text: str, tag: str) -> bytes:
    """A lossless model-messages blob carrying a tool call the entry tree lacks."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name=f"{_PLANTED_TOOL}_{tag}", args={}, tool_call_id=f"c-{tag}"),
                TextPart(content=final_text),
            ]
        ),
    ]
    return dump_model_messages(messages)


class _BlobRouter:
    """Fake Router emitting a caller-controlled blob + text per pass.

    ``blobs[i]`` is the ``FinalChunk.model_messages_json`` of the i-th
    ``stream_agentic`` call; the final text is ``resp<i+1>``. Every input
    ``message_history`` is recorded so a test can assert what the loop fed
    each pass.
    """

    def __init__(self, blobs: list[bytes | None]) -> None:
        self.stream_agentic_calls = 0
        self.histories: list[tuple[Any, ...]] = []
        self._blobs = blobs

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: Any = None,
        after_tool: Any = None,
        should_stop: Any = None,
    ) -> AsyncIterator[AgenticChunk]:
        del prompt, system, tools, toolsets, tier, before_tool, after_tool, should_stop
        index = self.stream_agentic_calls
        self.stream_agentic_calls += 1
        self.histories.append(tuple(message_history))
        yield ToolCallChunk(tool_name="live_tool", args_summary="")
        yield ToolResultChunk(tool_name="live_tool", result_summary="ok", ok=True)
        blob = self._blobs[index] if index < len(self._blobs) else None
        yield FinalChunk(text=f"resp{index + 1}", model_messages_json=blob)

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[Any],
        node_id: str = "",
        mcp_tools: tuple[McpToolSpec, ...] = (),
    ) -> Any:
        raise AssertionError("InteractiveLoop never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _loop(tmp_path: Path) -> InteractiveLoop:
    # hooks=None -> single-pass (today's behavior); compaction off for a
    # deterministic build_context().
    config = InteractiveLoopConfig(
        workspace_root=tmp_path, compaction=CompactionSettings(enabled=False)
    )
    return InteractiveLoop(config=config, hooks=None)


def _runtime(router: _BlobRouter, session: Session, tmp_path: Path) -> AgentRuntime:
    return AgentRuntime(
        session=session,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path / ".scratch"),
    )


async def _one_pass(
    loop: InteractiveLoop, runtime: AgentRuntime, user_input: str
) -> list[AgentEvent]:
    sink = AsyncIteratorEventSink()
    await loop.run(runtime=runtime, sink=sink, user_input=user_input)
    await sink.close()
    return [event async for event in sink]


def _first_user_entry_id(session: Session, content: str) -> str:
    for entry in session.path_to_root():
        if (
            isinstance(entry, MessageEntry)
            and entry.message.role == "user"
            and entry.message.content == content
        ):
            return entry.id
    raise AssertionError(f"no user entry with content {content!r} on the active branch")


async def test_ac005a_linear_continuation_reuses_lossless_blob(tmp_path: Path) -> None:
    """(a) A matching leaf stamp feeds the lossless blob, not a semantic rebuild."""
    blob_a = _blob_with_tool("first", "resp1", "a")
    router = _BlobRouter([blob_a, None])
    session = Session(storage=JsonlSessionStorage(tmp_path / "sess"), session_id="lin")
    runtime = _runtime(router, session, tmp_path)
    loop = _loop(tmp_path)

    await _one_pass(loop, runtime, "first")
    await _one_pass(loop, runtime, "second")

    # The second (linear) pass was fed the exact lossless blob from pass one.
    assert router.stream_agentic_calls == 2
    fed = dump_model_messages(router.histories[1])
    assert _PLANTED_TOOL.encode() in fed
    assert tuple(router.histories[1]) == load_model_messages(blob_a)
    # The planted tool detail is genuinely absent from a semantic rebuild —
    # proving the blob (not build_context()) supplied it.
    rebuilt = dump_model_messages(model_messages_from_messages(session.build_context()))
    assert _PLANTED_TOOL.encode() not in rebuilt


async def test_ac005b_branch_discards_stale_blob_and_reseeds(tmp_path: Path) -> None:
    """(b) A sibling branch discards the stale blob and reseeds from the tree."""
    blobs = [
        _blob_with_tool("first", "resp1", "a"),
        _blob_with_tool("second", "resp2", "b"),
        None,
    ]
    router = _BlobRouter(blobs)
    session = Session(storage=JsonlSessionStorage(tmp_path / "sess"), session_id="br")
    runtime = _runtime(router, session, tmp_path)
    loop = _loop(tmp_path)

    await _one_pass(loop, runtime, "first")
    await _one_pass(loop, runtime, "second")

    # Fork off the very first user turn; the latest blob belongs to the OTHER
    # branch and its leaf stamp cannot be on this branch's path.
    session.branch(_first_user_entry_id(session, "first"))
    await _one_pass(loop, runtime, "third")

    assert router.stream_agentic_calls == 3
    fed = router.histories[2]
    # The sibling branch's model history (its planted tool) is not inherited.
    assert _PLANTED_TOOL.encode() not in dump_model_messages(fed)
    # History is reseeded straight from the active branch's entry tree.
    expected = model_messages_from_messages(
        (
            Message(role="user", content="first"),
            Message(role="user", content="third"),
        )
    )
    assert tuple(fed) == tuple(expected)


async def test_ac005c_deleted_blob_reconstructs_from_entry_tree(tmp_path: Path) -> None:
    """(c) A missing ``messages.jsonl`` still reconstructs history from the tree."""
    blob_a = _blob_with_tool("first", "resp1", "a")
    router = _BlobRouter([blob_a, None])
    session_dir = tmp_path / "sess"
    session = Session(storage=JsonlSessionStorage(session_dir), session_id="del")
    runtime = _runtime(router, session, tmp_path)
    loop = _loop(tmp_path)

    await _one_pass(loop, runtime, "first")

    # Drop the lossless cache; the entry tree remains authoritative.
    blob_path = session_dir / MESSAGES_FILENAME
    assert blob_path.is_file()
    blob_path.unlink()

    events = await _one_pass(loop, runtime, "second")

    # No crash: the turn completes normally.
    assert isinstance(events[-1], LoopCompletedEvent)
    assert router.stream_agentic_calls == 2
    fed = router.histories[1]
    # With the blob gone, the planted tool cannot appear; history is the
    # semantic rebuild of the active branch.
    assert _PLANTED_TOOL.encode() not in dump_model_messages(fed)
    expected = model_messages_from_messages(
        (
            Message(role="user", content="first"),
            Message(role="assistant", content="resp1"),
            Message(role="user", content="second"),
        )
    )
    assert tuple(fed) == tuple(expected)
