"""Regression: phase-01 agent hook surface end-to-end via the public agent API.

Binding runtime example for spec ``plan-emergent-01-agent-hooks`` (ac-008).
A library-external user drives the offline ``TestModel`` through the public
``molexp.agent`` ``PydanticAIRouter`` re-export and the public
``molexp.agent.loops.hooks`` vocabulary only — no reach into ``_pydanticai``.

Checks (ac-008 a-d):
    (a) Supplied ``before_tool`` / ``after_tool`` hooks are invoked, each
        recording the dispatched tool's name (awaited at least once).
    (b) An ``after_tool`` returning ``HookOutcome.deny("blocked")`` flips the
        emitted ``ToolResultChunk.ok`` to ``False`` with ``"blocked"`` folded
        into ``result_summary``.
    (c) Omitting all three hooks reproduces the baseline ordered chunk-kind
        sequence a no-op proceed-returning triple produces (no behavior change).
    (d) ``HookOutcome`` factories carry the correct decision / message / token:
        ``proceed()`` empty, ``deny("m")`` message ``"m"``, ``suspend("t")``
        token ``"t"``.

Deterministic: offline ``TestModel`` only — no network, clock, or RNG. Run
standalone with ``python regressions/plan-emergent-01-agent-hooks.py``; the
final line on success is ``plan-emergent-01-agent-hooks: ok``.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from molexp.agent import PydanticAIRouter
from molexp.agent.loops.hooks import (
    HookDecision,
    HookOutcome,
    LoopState,
)
from molexp.agent.router import (
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)


async def peek(path: str) -> str:
    """A trivial read-only tool the offline model is expected to call."""
    return f"contents of {path}"


def _router() -> PydanticAIRouter:
    """Build a ``PydanticAIRouter`` whose every tier is an offline ``TestModel``."""
    return PydanticAIRouter(
        models={
            ModelTier.CHEAP: TestModel(),
            ModelTier.DEFAULT: TestModel(),
            ModelTier.HEAVY: TestModel(),
        },
    )


async def _check_a_hooks_invoked_with_tool_name() -> None:
    """ac-008(a): supplied hooks fire and record the dispatched tool name."""
    before_seen: list[tuple[str, object]] = []
    after_seen: list[tuple[str, str]] = []

    async def spy_before(*, tool_name: str, args: object) -> HookOutcome:
        before_seen.append((tool_name, args))
        return HookOutcome.proceed()

    async def spy_after(*, tool_name: str, result: str) -> HookOutcome:
        after_seen.append((tool_name, result))
        return HookOutcome.proceed()

    chunks = [
        chunk
        async for chunk in _router().stream_agentic(
            prompt="inspect",
            tools=(peek,),
            before_tool=spy_before,
            after_tool=spy_after,
        )
    ]

    tool_calls = [c for c in chunks if isinstance(c, ToolCallChunk)]
    assert tool_calls, "TestModel should have dispatched the tool"
    tool_name = tool_calls[0].tool_name

    assert before_seen, "before_tool must be awaited at least once"
    assert after_seen, "after_tool must be awaited at least once"
    assert tool_name in [name for name, _args in before_seen], (
        f"before_tool did not record {tool_name!r}"
    )
    assert tool_name in [name for name, _result in after_seen], (
        f"after_tool did not record {tool_name!r}"
    )


async def _check_b_deny_flips_ok() -> None:
    """ac-008(b): an after_tool deny flips ToolResultChunk.ok and folds the message."""
    deny_seen: list[tuple[str, str]] = []

    async def deny_after(*, tool_name: str, result: str) -> HookOutcome:
        deny_seen.append((tool_name, result))
        return HookOutcome.deny("blocked")

    chunks = [
        chunk
        async for chunk in _router().stream_agentic(
            prompt="inspect",
            tools=(peek,),
            after_tool=deny_after,
        )
    ]

    assert deny_seen, "after_tool must be awaited at least once"
    tool_results = [c for c in chunks if isinstance(c, ToolResultChunk)]
    assert tool_results, "expected at least one ToolResultChunk"
    assert all(chunk.ok is False for chunk in tool_results), (
        "deny must flip every ToolResultChunk.ok to False"
    )
    assert any("blocked" in chunk.result_summary for chunk in tool_results), (
        "deny message must be visible in result_summary"
    )


async def _check_c_no_op_hooks_preserve_sequence() -> None:
    """ac-008(c): omitting hooks reproduces the no-op proceed chunk-kind sequence."""

    async def noop_before(*, tool_name: str, args: object) -> HookOutcome:
        del tool_name, args
        return HookOutcome.proceed()

    async def noop_after(*, tool_name: str, result: str) -> HookOutcome:
        del tool_name, result
        return HookOutcome.proceed()

    async def noop_stop(*, state: LoopState) -> HookOutcome:
        del state
        return HookOutcome.proceed()

    baseline = [
        type(chunk).__name__
        async for chunk in _router().stream_agentic(prompt="inspect", tools=(peek,))
    ]
    hooked = [
        type(chunk).__name__
        async for chunk in _router().stream_agentic(
            prompt="inspect",
            tools=(peek,),
            before_tool=noop_before,
            after_tool=noop_after,
            should_stop=noop_stop,
        )
    ]

    assert baseline, "baseline stream produced no chunks"
    assert ToolCallChunk.__name__ in baseline, "baseline should exercise the tool path"
    assert baseline[-1] == FinalChunk.__name__, "stream must terminate with a FinalChunk"
    assert TextDeltaChunk.__name__ in baseline or FinalChunk.__name__ in baseline
    assert hooked == baseline, f"no-op hooks changed the chunk sequence: {hooked} != {baseline}"


def _check_d_outcome_payloads() -> None:
    """ac-008(d): HookOutcome proceed/deny/suspend carry the right message/token."""
    proceed = HookOutcome.proceed()
    assert proceed.decision == HookDecision.PROCEED
    assert proceed.message == ""
    assert proceed.token == ""
    assert proceed.is_proceed and not proceed.is_deny and not proceed.is_suspend

    deny = HookOutcome.deny("m")
    assert deny.decision == HookDecision.DENY
    assert deny.message == "m"
    assert deny.token == ""
    assert deny.is_deny and not deny.is_proceed and not deny.is_suspend

    suspend = HookOutcome.suspend("t")
    assert suspend.decision == HookDecision.SUSPEND
    assert suspend.token == "t"
    assert suspend.message == ""
    assert suspend.is_suspend and not suspend.is_proceed and not suspend.is_deny


async def main() -> int:
    """Run the four ac-008 checks; print the success marker as the final line."""
    await _check_a_hooks_invoked_with_tool_name()
    await _check_b_deny_flips_ok()
    await _check_c_no_op_hooks_preserve_sequence()
    _check_d_outcome_payloads()
    print("plan-emergent-01-agent-hooks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
