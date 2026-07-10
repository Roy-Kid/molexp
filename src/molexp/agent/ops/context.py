"""Build :class:`AgentSessionContext` for one InteractiveLoop turn."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.agent.ops.code_env import LocalCodeEnv
from molexp.agent.ops.discovery import CatalogDiscovery
from molexp.agent.ops.preamble import DefaultOpsBehavior
from molexp.agent.ops.protocols import AgentSessionContext, BehaviorPolicy, ToolSpec
from molexp.agent.ops.structure import WorkspaceStructureOps

if TYPE_CHECKING:
    from molexp.agent.execution_env import ExecutionEnv


def build_session_context(
    *,
    workspace_root: Path,
    execution_env: ExecutionEnv,
    mcp_toolsets: tuple[Any, ...] = (),
    mcp_tool_specs: tuple[ToolSpec, ...] = (),
    behavior: BehaviorPolicy | None = None,
) -> AgentSessionContext:
    """Assemble the three ops implementations + behavior for one turn."""
    root = Path(workspace_root).resolve()
    return AgentSessionContext(
        workspace_root=root,
        code=LocalCodeEnv(workspace_root=root, execution_env=execution_env),
        structure=WorkspaceStructureOps(root),
        discovery=CatalogDiscovery(
            workspace_root=root,
            mcp_toolsets=mcp_toolsets,
            mcp_tool_specs=mcp_tool_specs,
        ),
        behavior=behavior or DefaultOpsBehavior(),
    )
