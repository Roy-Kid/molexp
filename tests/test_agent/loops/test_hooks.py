"""SDK-free tool/hook vocabulary — ``molexp.agent.loops.hooks``.

The agent layer owns this domain-neutral hook vocabulary. Two behaviors are
worth locking here:

* :class:`HookOutcome`'s decision↔payload↔flag contract — the three factories
  (``proceed`` / ``deny`` / ``suspend``) route their payload into the right
  field and expose a matching ``is_*`` flag.
* The ``invoke_*`` helpers are the single source of the None→proceed honor
  convention: a missing hook proceeds, a real hook's outcome passes through
  verbatim.

Pydantic ``frozen`` enforcement, ``@runtime_checkable`` isinstance mechanics,
and field defaults are dependency/stdlib behavior and are not re-tested here.
"""

from __future__ import annotations

from molexp.agent.loops.hooks import (
    HookDecision,
    HookOutcome,
    LoopState,
    invoke_after_tool,
    invoke_before_tool,
    invoke_should_stop,
)


class TestHookOutcome:
    def test_factories_route_payload_by_decision(self) -> None:
        """proceed/deny/suspend set the decision, route payload, and flag is_*."""
        proceed = HookOutcome.proceed()
        assert proceed.decision == HookDecision.PROCEED
        assert (proceed.message, proceed.token) == ("", "")
        assert proceed.is_proceed and not proceed.is_deny and not proceed.is_suspend

        deny = HookOutcome.deny("no")
        assert deny.decision == HookDecision.DENY
        assert (deny.message, deny.token) == ("no", "")
        assert deny.is_deny and not deny.is_proceed and not deny.is_suspend

        suspend = HookOutcome.suspend("tok")
        assert suspend.decision == HookDecision.SUSPEND
        assert (suspend.message, suspend.token) == ("", "tok")
        assert suspend.is_suspend and not suspend.is_proceed and not suspend.is_deny


class TestInvokeHelpers:
    async def test_invoke_before_tool_honors_none_as_proceed(self) -> None:
        outcome = await invoke_before_tool(None, tool_name="t", args={})
        assert outcome == HookOutcome.proceed()

    async def test_invoke_after_tool_honors_none_as_proceed(self) -> None:
        outcome = await invoke_after_tool(None, tool_name="t", result="")
        assert outcome == HookOutcome.proceed()

    async def test_invoke_should_stop_honors_none_as_proceed(self) -> None:
        outcome = await invoke_should_stop(None, state=LoopState(step=1))
        assert outcome == HookOutcome.proceed()

    async def test_invoke_before_tool_passes_hook_outcome_through_verbatim(self) -> None:
        """A real hook's outcome (decision + token) flows through unchanged."""

        async def suspend_hook(*, tool_name: str, args: object) -> HookOutcome:
            return HookOutcome.suspend("t")

        outcome = await invoke_before_tool(suspend_hook, tool_name="tool", args={})
        assert outcome.decision == HookDecision.SUSPEND
        assert outcome.token == "t"
