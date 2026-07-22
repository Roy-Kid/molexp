"""SDK-free tool/hook vocabulary — ``molexp.agent.loops.hooks`` (ac-001..ac-004).

Pins the domain-neutral hook primitives Phase 01 introduces:

* ac-001 — :class:`HookOutcome` factories (``proceed`` / ``deny`` / ``suspend``),
  their ``is_*`` convenience properties, and frozen immutability.
* ac-002 — the three ``@runtime_checkable`` Protocols
  (:class:`BeforeToolHook` / :class:`AfterToolHook` / :class:`ShouldStopGuard`)
  accept a conforming async callable and reject a non-conforming object.
* ac-003 — the ``invoke_*`` helpers honor ``None`` by returning
  :meth:`HookOutcome.proceed`.
* ac-004 — the ``invoke_*`` helpers pass a real hook's outcome through
  verbatim (deny / suspend).

Also pins :class:`LoopState` basics (constructor defaults + frozen), since the
guard helper consumes it. These tests are RED until
``src/molexp/agent/loops/hooks.py`` exists — the module import fails first.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent.loops.hooks import (
    AfterToolHook,
    BeforeToolHook,
    HookDecision,
    HookOutcome,
    LoopState,
    ShouldStopGuard,
    invoke_after_tool,
    invoke_before_tool,
    invoke_should_stop,
)

# ── ac-001 · HookOutcome factories + properties + frozen ────────────────────


def test_hook_outcome_proceed_factory() -> None:
    """``proceed()`` → PROCEED decision, empty message/token, is_proceed."""
    outcome = HookOutcome.proceed()
    assert outcome.decision == HookDecision.PROCEED
    assert outcome.message == ""
    assert outcome.token == ""
    assert outcome.is_proceed
    assert not outcome.is_deny
    assert not outcome.is_suspend


def test_hook_outcome_deny_factory() -> None:
    """``deny("no")`` → DENY decision carrying the message."""
    outcome = HookOutcome.deny("no")
    assert outcome.decision == HookDecision.DENY
    assert outcome.message == "no"
    assert outcome.is_deny
    assert not outcome.is_proceed
    assert not outcome.is_suspend


def test_hook_outcome_suspend_factory() -> None:
    """``suspend("tok")`` → SUSPEND decision carrying the token."""
    outcome = HookOutcome.suspend("tok")
    assert outcome.decision == HookDecision.SUSPEND
    assert outcome.token == "tok"
    assert outcome.is_suspend
    assert not outcome.is_proceed
    assert not outcome.is_deny


def test_hook_outcome_is_frozen() -> None:
    """Assigning to a :class:`HookOutcome` field raises (frozen model)."""
    outcome = HookOutcome.proceed()
    with pytest.raises(ValidationError):
        outcome.decision = HookDecision.DENY  # type: ignore[misc]


# ── ac-002 · runtime_checkable hook Protocols ───────────────────────────────


def test_before_tool_hook_is_runtime_checkable() -> None:
    """A conforming async callable is a ``BeforeToolHook``; a plain object is not."""

    class _Before:
        async def __call__(self, *, tool_name: str, args: object) -> HookOutcome:
            return HookOutcome.proceed()

    assert isinstance(_Before(), BeforeToolHook)
    assert not isinstance(object(), BeforeToolHook)


def test_after_tool_hook_is_runtime_checkable() -> None:
    """A conforming async callable is an ``AfterToolHook``; a plain object is not."""

    class _After:
        async def __call__(self, *, tool_name: str, result: str) -> HookOutcome:
            return HookOutcome.proceed()

    assert isinstance(_After(), AfterToolHook)
    assert not isinstance(object(), AfterToolHook)


def test_should_stop_guard_is_runtime_checkable() -> None:
    """A conforming async callable is a ``ShouldStopGuard``; a plain object is not."""

    class _Stop:
        async def __call__(self, *, state: LoopState) -> HookOutcome:
            return HookOutcome.proceed()

    assert isinstance(_Stop(), ShouldStopGuard)
    assert not isinstance(object(), ShouldStopGuard)


# ── ac-003 · invoke_* honor None with proceed ───────────────────────────────


async def test_invoke_before_tool_none_returns_proceed() -> None:
    """``invoke_before_tool(None, ...)`` returns ``HookOutcome.proceed()``."""
    outcome = await invoke_before_tool(None, tool_name="t", args={})
    assert outcome == HookOutcome.proceed()


async def test_invoke_after_tool_none_returns_proceed() -> None:
    """``invoke_after_tool(None, ...)`` returns ``HookOutcome.proceed()``."""
    outcome = await invoke_after_tool(None, tool_name="t", result="")
    assert outcome == HookOutcome.proceed()


async def test_invoke_should_stop_none_returns_proceed() -> None:
    """``invoke_should_stop(None, ...)`` returns ``HookOutcome.proceed()``."""
    outcome = await invoke_should_stop(None, state=LoopState(step=1))
    assert outcome == HookOutcome.proceed()


# ── ac-004 · invoke_* pass a real hook's outcome through verbatim ────────────


async def test_invoke_before_tool_propagates_deny() -> None:
    """A ``deny`` hook makes ``invoke_before_tool`` return that outcome verbatim."""

    async def deny_hook(*, tool_name: str, args: object) -> HookOutcome:
        return HookOutcome.deny("x")

    outcome = await invoke_before_tool(deny_hook, tool_name="t", args={})
    assert outcome.decision == HookDecision.DENY
    assert outcome.message == "x"


async def test_invoke_before_tool_propagates_suspend() -> None:
    """A ``suspend`` hook propagates decision + token through ``invoke_before_tool``."""

    async def suspend_hook(*, tool_name: str, args: object) -> HookOutcome:
        return HookOutcome.suspend("t")

    outcome = await invoke_before_tool(suspend_hook, tool_name="tool", args={})
    assert outcome.decision == HookDecision.SUSPEND
    assert outcome.token == "t"


# ── LoopState basics (constructor defaults + frozen) ────────────────────────


def test_loop_state_defaults() -> None:
    """``LoopState`` fills ``last_tool`` / ``draft_final`` with empty defaults."""
    state = LoopState(step=3)
    assert state.step == 3
    assert state.last_tool == ""
    assert state.draft_final == ""


def test_loop_state_is_frozen() -> None:
    """Assigning to a :class:`LoopState` field raises (frozen model)."""
    state = LoopState(step=1)
    with pytest.raises(ValidationError):
        state.step = 2  # type: ignore[misc]
