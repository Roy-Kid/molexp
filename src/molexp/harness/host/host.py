"""``Host`` — compose, load, unload a plugin tree."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.harness.errors import PluginInjectError
from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.host.plugin import Plugin

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext

__all__ = ["Host"]


class Host:
    """One running plugin tree.

    ``mount`` waits on ``plugin.inject``, then ``apply``. ``unload``
    disposes effects in reverse (one plugin, or the whole tree).
    """

    def __init__(self) -> None:
        self.ctx = Context()
        self._stack: list[tuple[str, int]] = []

    def dump(self) -> list[str]:
        """Plugin names in mount order."""
        return [name for name, _ in self._stack]

    def dump_config(self) -> dict[str, object]:
        """What this process actually booted: plugins + published keys."""
        return {
            "plugins": self.dump(),
            "services": list(self.ctx.service_keys()),
        }

    def mount(self, plugin: Plugin) -> None:
        """Activate *plugin*. Missing inject keys fail loud."""
        missing = tuple(key for key in plugin.inject if not self.ctx.has(key))
        if missing:
            raise PluginInjectError(plugin.name, missing)
        mark = self.ctx.effect_count()
        plugin.apply(self.ctx)
        self._stack.append((plugin.name, mark))

    def unload(self, name: str | None = None) -> None:
        """Unload *name* and everything mounted after it, or the whole tree."""
        if not self._stack:
            return
        if name is None:
            self.ctx.unwind_to(0)
            self._stack.clear()
            return
        index = None
        for i, (mounted, _) in enumerate(self._stack):
            if mounted == name:
                index = i
        if index is None:
            raise KeyError(f"plugin {name!r} is not mounted")
        mark = self._stack[index][1]
        self.ctx.unwind_to(mark)
        del self._stack[index:]

    def as_run_context(self) -> HarnessRunContext:
        """Project host services onto the Stage-facing container."""
        from molexp.harness.core.run_context import HarnessRunContext
        from molexp.harness.gateways.gateway import AgentGateway
        from molexp.harness.registry.capability_registry import CapabilityRegistry
        from molexp.harness.store.approval_store import ApprovalStore
        from molexp.harness.store.artifact_store import ArtifactStore
        from molexp.harness.store.event_log import EventLog
        from molexp.harness.store.lineage_store import ArtifactLineageStore

        ctx = self.ctx
        artifacts = ctx.require(Keys.ARTIFACTS)
        events = ctx.require(Keys.EVENTS)
        lineage = ctx.require(Keys.LINEAGE)
        if not isinstance(artifacts, ArtifactStore):
            raise TypeError(f"{Keys.ARTIFACTS} is not an ArtifactStore")
        if not isinstance(events, EventLog):
            raise TypeError(f"{Keys.EVENTS} is not an EventLog")
        if not isinstance(lineage, ArtifactLineageStore):
            raise TypeError(f"{Keys.LINEAGE} is not an ArtifactLineageStore")

        capabilities = ctx.get(Keys.CAPABILITIES)
        if capabilities is not None and not isinstance(capabilities, CapabilityRegistry):
            raise TypeError(f"{Keys.CAPABILITIES} is not a CapabilityRegistry")
        gateway = ctx.get(Keys.LLM)
        if gateway is not None and not isinstance(gateway, AgentGateway):
            raise TypeError(f"{Keys.LLM} is not an AgentGateway")
        approval = ctx.get(Keys.APPROVAL)
        if approval is not None and not isinstance(approval, ApprovalStore):
            raise TypeError(f"{Keys.APPROVAL} is not an ApprovalStore")

        return HarnessRunContext(
            run_id=str(ctx.require(Keys.RUN_ID)),
            workspace_root=Path(str(ctx.require(Keys.WORKSPACE_ROOT))),
            artifact_store=artifacts,
            event_log=events,
            lineage_store=lineage,
            capability_registry=capabilities,
            agent_gateway=gateway,
            approval_store=approval,
        )
