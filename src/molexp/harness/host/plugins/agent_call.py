"""AgentCall plugin — the harness atom, with host waterfalls around ``call``."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from molexp.harness.errors import AgentCallRejectedError
from molexp.harness.gateways.call_runtime import AgentCallRuntime
from molexp.harness.gateways.gateway import AgentGateway
from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.schemas import AgentCallResult, AgentCallSpec
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["AgentCallGateway", "AgentCallPlugin", "AgentStep"]


@dataclass(frozen=True, slots=True)
class AgentStep:
    """``agent/post-step`` payload: the spec that ran and its result."""

    spec: AgentCallSpec
    result: AgentCallResult


class AgentCallGateway:
    """Provider: wrap an :class:`AgentGateway` in ``agent/pre-step`` / post-step."""

    def __init__(self, inner: AgentGateway, ctx: Context) -> None:
        self._inner = inner
        self._ctx = ctx

    @property
    def router(self) -> object:
        """Forward the inner router's public accessor (plan/chat runners)."""
        return getattr(self._inner, "router", None)

    async def call(
        self,
        spec: AgentCallSpec,
        *,
        runtime: AgentCallRuntime | None = None,
    ) -> AgentCallResult:
        """One AgentCall: ``agent/pre-step`` waterfall, then the inner gateway, then emit."""
        rewritten = await self._ctx.waterfall("agent/pre-step", spec)
        if not isinstance(rewritten, AgentCallSpec):
            raise AgentCallRejectedError(
                "agent/pre-step must return an AgentCallSpec; reject by raising"
            )
        params = inspect.signature(self._inner.call).parameters
        if "runtime" in params:
            result = await self._inner.call(rewritten, runtime=runtime)
        elif runtime is not None:
            raise TypeError("inner AgentGateway.call does not accept runtime=")
        else:
            result = await self._inner.call(rewritten)
        posted = await self._ctx.waterfall(
            "agent/post-step",
            AgentStep(spec=rewritten, result=result),
        )
        if isinstance(posted, AgentStep):
            return posted.result
        if isinstance(posted, AgentCallResult):
            return posted
        raise TypeError("agent/post-step must return an AgentStep or AgentCallResult")


class AgentCallPlugin:
    """Publish the AgentCall atom as ``ctx.llm``.

    Injects :data:`Keys.ARTIFACTS` and, when the inner gateway supports it,
    rebinds persist onto that store so the host and the gateway share one
    object (not two ``FileArtifactStore`` s on the same directory).
    """

    name = "llm"
    inject: tuple[str, ...] = (Keys.ARTIFACTS,)

    def __init__(self, gateway: AgentGateway) -> None:
        self._gateway = gateway

    def apply(self, ctx: Context) -> None:
        """Wrap *gateway*, bind persist, publish :data:`Keys.LLM`."""
        store = ctx.require(Keys.ARTIFACTS)
        bind = getattr(self._gateway, "bind_artifact_store", None)
        if callable(bind) and isinstance(store, ArtifactStore):
            bind(store)
        ctx.provide(Keys.LLM, AgentCallGateway(self._gateway, ctx))
