"""PydanticAIRuntime: concrete AgentRuntime implementation.

Wraps pydantic-ai Agent with:
- MolexpToolCatalog (built-in + user tools)
- ApprovalPolicy via pydantic-ai's approval_required()
- Session persistence in workspace sessions/ directory
- Message history for cross-process resumption
"""

from __future__ import annotations

import json
from mollog import get_logger
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from ..policy import ApprovalPolicy
from ..runtime import AgentRuntime
from ..tools import Tool
from ..types import (
    AgentSession,
    Goal,
    ToolContext,  # noqa: F401 - re-exported for tool wrappers
)
from .catalog import MolexpToolCatalog
from .deps import MolexpDeps
from .session import PydanticAISession

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a research experiment assistant integrated with the molexp workspace.

Your role:
1. Understand the user's research goal and any constraints
2. Plan a sequence of steps to achieve the goal
3. Use the available tools to explore the workspace, create runs, and execute workflows
4. Observe results and adjust your plan as needed
5. Report clearly when the goal is achieved or when it cannot be met

Available tool levels:
- Workspace tools: list and inspect projects, experiments, runs (read-only)
- Product tools: create new runs with specified parameters (write)
- Workflow tools: execute workflows (available in Phase 3)

Always start by exploring the workspace to understand the current state before taking actions.
When creating runs, specify meaningful parameters that align with the research goal.
"""


class PydanticAIRuntime(AgentRuntime):
    """AgentRuntime backed by pydantic-ai.

    Creates a pydantic-ai Agent configured with the MolexpToolCatalog
    and manages the session lifecycle including persistence.

    Args:
        model: pydantic-ai model name or instance (default: claude-sonnet-4-6)
        sessions_dir_name: Workspace subdirectory for session persistence
    """

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        sessions_dir_name: str = "sessions",
    ) -> None:
        self._model = model
        self._sessions_dir_name = sessions_dir_name
        self._active_sessions: dict[str, PydanticAISession] = {}

    def _get_sessions_dir(self, workspace: Any) -> Path:
        sessions_dir = Path(workspace.root) / self._sessions_dir_name
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir

    def _build_agent(
        self,
        extra_tools: list[Tool],
        approval_policy: ApprovalPolicy,
    ) -> Agent[MolexpDeps, str]:
        catalog = MolexpToolCatalog(
            extra_tools=extra_tools,
            approval_policy=approval_policy,
        )
        toolset = catalog.build()

        return Agent(
            model=self._model,
            system_prompt=_SYSTEM_PROMPT,
            deps_type=MolexpDeps,
            toolsets=[toolset],
        )

    def _goal_to_prompt(self, goal: Goal) -> str:
        lines = [f"Goal: {goal.description}"]
        if goal.constraints:
            lines.append(f"Constraints: {goal.constraints}")
        if goal.success_criteria:
            lines.append("Success criteria:")
            for criterion in goal.success_criteria:
                lines.append(f"  - {criterion}")
        return "\n".join(lines)

    async def start_session(
        self,
        goal: Goal,
        workspace: Any,
        extra_tools: list[Tool],
        approval_policy: ApprovalPolicy,
    ) -> AgentSession:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"

        session = PydanticAISession(
            session_id=session_id,
            goal=goal,
            workspace=workspace,
        )

        deps = MolexpDeps(
            workspace=workspace,
            session_id=session_id,
            session=session,
        )

        agent = self._build_agent(extra_tools, approval_policy)
        prompt = self._goal_to_prompt(goal)

        # Register before launching so persistence can find it
        self._active_sessions[session_id] = session

        # Persist session metadata
        self._save_session_metadata(session, workspace)

        # Launch agent run as background task
        session._launch(agent=agent, prompt=prompt, deps=deps)

        logger.info(f"Started agent session {session_id}")
        return session

    async def resume_session(
        self,
        session_id: str,
        workspace: Any,
    ) -> AgentSession:
        sessions_dir = self._get_sessions_dir(workspace)
        session_dir = sessions_dir / session_id

        if not session_dir.exists():
            raise ValueError(f"Session '{session_id}' not found in workspace")

        # Load metadata
        meta_path = session_dir / "metadata.json"
        with meta_path.open() as f:
            meta = json.load(f)

        goal = Goal(
            description=meta["goal"]["description"],
            constraints=meta["goal"].get("constraints", {}),
            success_criteria=meta["goal"].get("success_criteria", []),
        )

        session = PydanticAISession(
            session_id=session_id,
            goal=goal,
            workspace=workspace,
        )

        # Restore message history if available
        history_path = session_dir / "history.json"
        if history_path.exists():
            from pydantic_ai.messages import ModelMessagesTypeAdapter
            with history_path.open("rb") as f:
                history = ModelMessagesTypeAdapter.validate_json(f.read())
            session.restore_message_history(history)

        # Re-launch with empty extra_tools and default approval policy
        # (user can pass updated tools via AgentService.resume)
        deps = MolexpDeps(
            workspace=workspace,
            session_id=session_id,
            session=session,
        )
        agent = self._build_agent([], ApprovalPolicy())
        prompt = "Resume from where we left off and continue towards the original goal."

        self._active_sessions[session_id] = session
        session._launch(agent=agent, prompt=prompt, deps=deps)

        logger.info(f"Resumed agent session {session_id}")
        return session

    async def get_session_history(self, session_id: str) -> Any:
        session = self._active_sessions.get(session_id)
        if session is not None:
            return {"session_id": session_id, "messages": session.get_message_history()}
        return {"session_id": session_id, "messages": []}

    def _save_session_metadata(self, session: PydanticAISession, workspace: Any) -> None:
        sessions_dir = self._get_sessions_dir(workspace)
        session_dir = sessions_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_id": session.session_id,
            "status": session.status,
            "goal": {
                "description": session.goal.description,
                "constraints": session.goal.constraints,
                "success_criteria": session.goal.success_criteria,
            },
        }
        meta_path = session_dir / "metadata.json"
        tmp_path = meta_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(meta, indent=2))
        tmp_path.rename(meta_path)

    def save_session_history(self, session: PydanticAISession, workspace: Any) -> None:
        """Persist message history for resumption (call after session completes)."""
        try:
            from pydantic_ai.messages import ModelMessagesTypeAdapter
            history = session.get_message_history()
            if not history:
                return
            sessions_dir = self._get_sessions_dir(workspace)
            session_dir = sessions_dir / session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            history_path = session_dir / "history.json"
            tmp_path = history_path.with_suffix(".tmp")
            tmp_path.write_bytes(
                ModelMessagesTypeAdapter.dump_json(history, indent=2)
            )
            tmp_path.rename(history_path)
        except Exception:
            logger.exception(f"Failed to save session history for {session.session_id}")
