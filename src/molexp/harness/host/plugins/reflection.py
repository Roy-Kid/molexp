"""``Reflection`` — OOP plugin on ``agent/post-step``. Not an Agent subclass."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.host.plugins.agent_call import AgentStep
from molexp.harness.schemas import AgentCallResult

__all__ = ["Reflection"]

Critic = Callable[[AgentStep], Awaitable[AgentCallResult | None]]


class Reflection:
    """After each AgentCall, optionally replace the result via *critic*."""

    name = "reflection"
    inject: tuple[str, ...] = (Keys.LLM,)

    def __init__(
        self,
        *,
        critic: Critic | None = None,
        skip: tuple[str, ...] = ("reflect",),
    ) -> None:
        self.critic = critic
        self.skip = skip

    def apply(self, ctx: Context) -> None:
        """Subscribe to ``agent/post-step``."""

        async def on_post(value: object, nxt: Callable[..., Awaitable[object]]) -> object:
            current = await nxt(value)
            if self.critic is None or not isinstance(current, AgentStep):
                return current
            if current.spec.agent_name in self.skip:
                return current
            revised = await self.critic(current)
            if revised is None:
                return current
            if not isinstance(revised, AgentCallResult):
                raise TypeError("Reflection critic must return AgentCallResult or None")
            return AgentStep(spec=current.spec, result=revised)

        ctx.on("agent/post-step", on_post, mode="waterfall")
