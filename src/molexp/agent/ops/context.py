"""Build :class:`AgentSessionContext` for one ReAct turn."""

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
    confine_code_to: str | None = None,
    surface: str = "chat",
) -> AgentSessionContext:
    """Assemble the three ops implementations + behavior for one turn.

    Args:
        confine_code_to: When set (Chat Mode: ``agent/.scratch``),
            :class:`LocalCodeEnv` rewrites writes under that prefix.
        surface: Builtin catalog preset forwarded to
            :class:`~molexp.agent.ops.discovery.CatalogDiscovery`.
    """
    root = Path(workspace_root).resolve()
    return AgentSessionContext(
        workspace_root=root,
        code=LocalCodeEnv(
            workspace_root=root,
            execution_env=execution_env,
            confine_to=confine_code_to,
        ),
        structure=WorkspaceStructureOps(root),
        discovery=CatalogDiscovery(
            workspace_root=root,
            mcp_toolsets=mcp_toolsets,
            mcp_tool_specs=mcp_tool_specs,
            builtin_surface=surface,
        ),
        behavior=behavior or DefaultOpsBehavior(),
    )
