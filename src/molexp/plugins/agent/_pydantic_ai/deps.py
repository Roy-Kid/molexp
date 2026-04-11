"""MolexpDeps: dependency injection container for pydantic-ai agent runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MolexpDeps:
    """Dependencies injected into every tool call via RunContext[MolexpDeps].

    Attributes:
        workspace: Root Workspace instance for project/experiment/run access
        session_id: ID of the current agent session
        session: PydanticAISession reference (for tool → ToolContext bridge)
        current_run: Active Run if a workflow is currently executing
    """

    workspace: Any
    session_id: str
    session: Any = None
    current_run: Any = None
