"""``stream_agentic`` hook threading — protocol + impl (ac-005..ac-007).

Phase 01 threads three optional, keyword-only hook params
(``before_tool`` / ``after_tool`` / ``should_stop``) into ``stream_agentic``
on both the SDK-free :class:`~molexp.agent.router.Router` Protocol and the
concrete :class:`~molexp.agent._pydanticai.router.PydanticAIRouter`:

* ac-005 — both signatures expose the three params as ``KEYWORD_ONLY`` with a
  ``None`` default.
* ac-006 — no-behavior-change invariant: omitting the hooks vs. supplying
  no-op proceed-returning hooks yields the *same ordered sequence of chunk
  types* (offline ``TestModel`` + one tool).
* ac-007 — hooks are invoked and honored: spy ``before_tool`` / ``after_tool``
  are each ``await``ed at least once and record the ``tool_name``; an
  ``after_tool`` returning ``deny("blocked")`` flips the emitted
  :class:`~molexp.agent.router.ToolResultChunk` ``ok`` to ``False`` and makes
  ``"blocked"`` visible in ``result_summary``.

RED until the hook params exist: the module-level import of
``molexp.agent.loops.hooks`` fails first, and ac-005 asserts params the current
signatures lack.
"""

from __future__ import annotations

import inspect

import pytest

from molexp.agent.loops.hooks import HookOutcome, LoopState
from molexp.agent.router import (
    Router,
    ToolCallChunk,
    ToolResultChunk,
)

pytestmark = pytest.mark.asyncio


def _router(model: object) -> object:
    """Build a :class:`PydanticAIRouter` whose every tier is ``model``."""
    from molexp.agent._pydanticai.router import PydanticAIRouter
    from molexp.agent.router import ModelTier

    return PydanticAIRouter(
        models={
            ModelTier.CHEAP: model,
            ModelTier.DEFAULT: model,
            ModelTier.HEAVY: model,
        },
    )


def _assert_kwonly_none(sig: inspect.Signature, name: str) -> None:
    """Assert ``sig`` has a keyword-only param ``name`` defaulting to ``None``."""
    assert name in sig.parameters, f"{name!r} missing from {sig}"
    param = sig.parameters[name]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{name!r} is not keyword-only"
    assert param.default is None, f"{name!r} default is {param.default!r}, expected None"


# ── ac-005 · signatures expose the three hook params ────────────────────────


async def test_stream_agentic_signatures_expose_hook_params() -> None:
    """Both Protocol and impl ``stream_agentic`` expose the kw-only hook params."""
    pytest.importorskip("pydantic_ai")
    from molexp.agent._pydanticai.router import PydanticAIRouter

    targets = (Router.stream_agentic, PydanticAIRouter.stream_agentic)
    for target in targets:
        sig = inspect.signature(target)
        for name in ("before_tool", "after_tool", "should_stop"):
            _assert_kwonly_none(sig, name)


# ── ac-006 · no behavior change when hooks are no-ops ────────────────────────


async def test_stream_agentic_no_op_hooks_preserve_chunk_sequence() -> None:
    """No-op proceed hooks yield the same ordered chunk-type sequence as omitting them."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    async def peek(path: str) -> str:
        """A trivial read-only tool the model may call."""
        return f"contents of {path}"

    async def noop_before(*, tool_name: str, args: object) -> HookOutcome:
        return HookOutcome.proceed()

    async def noop_after(*, tool_name: str, result: str) -> HookOutcome:
        return HookOutcome.proceed()

    async def noop_stop(*, state: LoopState) -> HookOutcome:
        return HookOutcome.proceed()

    baseline_router = _router(TestModel())
    baseline = [
        type(chunk)
        async for chunk in baseline_router.stream_agentic(prompt="inspect", tools=(peek,))
    ]

    hooked_router = _router(TestModel())
    hooked = [
        type(chunk)
        async for chunk in hooked_router.stream_agentic(
            prompt="inspect",
            tools=(peek,),
            before_tool=noop_before,
            after_tool=noop_after,
            should_stop=noop_stop,
        )
    ]

    assert hooked == baseline


# ── ac-007 · hooks are invoked and honored ──────────────────────────────────


async def test_stream_agentic_invokes_and_honors_hooks() -> None:
    """Spy hooks fire and record; an ``after_tool`` deny flips ToolResultChunk.ok."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    async def peek(path: str) -> str:
        """A trivial read-only tool the model may call."""
        return f"contents of {path}"

    before_calls: list[str] = []
    after_calls: list[str] = []

    async def spy_before(*, tool_name: str, args: object) -> HookOutcome:
        before_calls.append(tool_name)
        return HookOutcome.proceed()

    async def deny_after(*, tool_name: str, result: str) -> HookOutcome:
        after_calls.append(tool_name)
        return HookOutcome.deny("blocked")

    router = _router(TestModel())
    chunks = [
        chunk
        async for chunk in router.stream_agentic(
            prompt="inspect",
            tools=(peek,),
            before_tool=spy_before,
            after_tool=deny_after,
        )
    ]

    tool_calls = [chunk for chunk in chunks if isinstance(chunk, ToolCallChunk)]
    assert tool_calls, "TestModel should have dispatched the tool"
    expected_name = tool_calls[0].tool_name

    assert before_calls, "before_tool must be awaited at least once"
    assert after_calls, "after_tool must be awaited at least once"
    assert expected_name in before_calls
    assert expected_name in after_calls

    tool_results = [chunk for chunk in chunks if isinstance(chunk, ToolResultChunk)]
    assert tool_results, "expected at least one ToolResultChunk"
    assert all(not chunk.ok for chunk in tool_results)
    assert any("blocked" in chunk.result_summary for chunk in tool_results)
