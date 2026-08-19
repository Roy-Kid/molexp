"""Profile composers — named plugin stacks on a :class:`Host`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.harness.host.host import Host
from molexp.harness.host.plugins.agent_call import AgentCallPlugin
from molexp.harness.host.plugins.capabilities import CapabilitiesPlugin
from molexp.harness.host.plugins.executor import ExecutorPlugin
from molexp.harness.host.plugins.stores import RunStoresPlugin
from molexp.harness.host.plugins.tools import ToolsPlugin
from molexp.harness.host.plugins.workflow import WorkflowPlugin

if TYPE_CHECKING:
    from molexp.harness.executors import Executor
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.host.plugin import Plugin
    from molexp.harness.registry.capability_registry import CapabilityRegistry

__all__ = ["compose_chat", "compose_curate", "compose_plan", "compose_run"]


def _mount_extra(host: Host, extra: tuple[Plugin, ...]) -> None:
    for plugin in extra:
        host.mount(plugin)


def compose_chat(
    *,
    gateway: AgentGateway,
    scratch_dir: Path,
    extra: tuple[Plugin, ...] = (),
) -> Host:
    """Chat profile: scratch stores + tools belt + AgentCall."""
    host = Host()
    host.mount(
        RunStoresPlugin(
            run_id="chat",
            run_dir=Path(scratch_dir),
            workspace_root=Path(scratch_dir),
        )
    )
    host.mount(ToolsPlugin())
    host.mount(AgentCallPlugin(gateway))
    _mount_extra(host, extra)
    return host


def compose_plan(
    *,
    run_id: str,
    run_dir: Path,
    gateway: AgentGateway,
    capability_registry: CapabilityRegistry | None = None,
    workspace_root: Path | None = None,
    extra: tuple[Plugin, ...] = (),
) -> Host:
    """``plan`` bundle: run stores + tools + workspace + workflow + llm."""
    from molexp.workspace.plugin import WorkspacePlugin

    host = Host()
    root = Path(workspace_root) if workspace_root is not None else Path(run_dir)
    host.mount(
        RunStoresPlugin(
            run_id=run_id,
            run_dir=Path(run_dir),
            workspace_root=workspace_root,
        )
    )
    if capability_registry is not None:
        host.mount(CapabilitiesPlugin(capability_registry))
    host.mount(ToolsPlugin())
    host.mount(WorkspacePlugin(root))
    host.mount(WorkflowPlugin())
    host.mount(AgentCallPlugin(gateway))
    _mount_extra(host, extra)
    return host


def compose_curate(
    *,
    run_id: str,
    run_dir: Path,
    workspace_root: Path,
    gateway: AgentGateway | None = None,
    capability_registry: CapabilityRegistry | None = None,
    extra: tuple[Plugin, ...] = (),
) -> Host:
    """``curate`` bundle: run-local stores; Stage root is the real workspace."""
    host = Host()
    host.mount(
        RunStoresPlugin(
            run_id=run_id,
            run_dir=Path(run_dir),
            workspace_root=Path(workspace_root),
        )
    )
    if capability_registry is not None:
        host.mount(CapabilitiesPlugin(capability_registry))
    if gateway is not None:
        host.mount(ToolsPlugin())
        host.mount(AgentCallPlugin(gateway))
    _mount_extra(host, extra)
    return host


def compose_run(
    *,
    run_id: str,
    run_dir: Path,
    workspace_root: Path | None = None,
    executor: Executor | None = None,
    extra: tuple[Plugin, ...] = (),
) -> Host:
    """``run`` bundle: stores + executor + workspace + workflow. Science extras last."""
    from molexp.workspace.plugin import WorkspacePlugin

    host = Host()
    root = Path(workspace_root) if workspace_root is not None else Path(run_dir)
    host.mount(
        RunStoresPlugin(
            run_id=run_id,
            run_dir=Path(run_dir),
            workspace_root=workspace_root,
        )
    )
    host.mount(ExecutorPlugin(executor))
    host.mount(WorkspacePlugin(root))
    host.mount(WorkflowPlugin())
    _mount_extra(host, extra)
    return host
