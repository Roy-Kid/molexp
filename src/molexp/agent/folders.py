"""Agent-layer Concept types — ``Agent`` + ``AgentSession``.

Rehomed onto :class:`molexp.knowledge.Folder` (the OKF rewrite): an ``Agent``
is a knowledge Concept (``type = "agent.agent"``) whose ``AgentSession``
children (``type = "agent.session"``) are **flat** child Concepts — one dir per
session, ``meta.yaml`` for structured state, ``messages.jsonl`` for the
pydantic-ai history. Both register with the knowledge concept-type registry, so
``add_folder(name, concept_type="agent.session")`` / ``walk`` / ``get`` rebuild
the right subclass. All I/O routes through the Concept's injectable
``FileSystem`` (``self._fs``), so a session works against any backend.

The *runtime* ``AgentSession`` (in :mod:`molexp.agent.session`) is a distinct
in-memory class; this one is its on-disk persistent counterpart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from molexp.agent.folders_metadata import AgentMeta, AgentSessionMeta, SessionStatusStr
from molexp.knowledge import FileSystem, Folder, concept_type

AGENT_KIND = "agent.agent"
AGENT_SESSION_KIND = "agent.session"
MESSAGES_FILENAME = "messages.jsonl"


@concept_type(AGENT_SESSION_KIND)
class AgentSession(Folder):
    """One conversation under an :class:`Agent` — ``type = "agent.session"``.

    Holds ``meta.yaml`` (:class:`AgentSessionMeta`) + ``messages.jsonl`` (the
    pydantic-ai history, written via :meth:`write_messages` through ``self._fs``).
    """

    DEFAULT_TYPE = AGENT_SESSION_KIND

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
        fs: FileSystem | None = None,
        goal_summary: str = "",
        status: SessionStatusStr = "pending",
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
            fs=fs,
        )
        self._meta = AgentSessionMeta(id=self._name, goal_summary=goal_summary, status=status)

    # ── typed meta (meta.yaml) ─────────────────────────────────────────────

    def read_session_meta(self) -> AgentSessionMeta:
        """Load this session's typed ``meta.yaml`` from disk (disk is truth)."""
        return AgentSessionMeta.model_validate(self.read_meta().model_dump())

    def write_session_meta(self, meta: AgentSessionMeta) -> None:
        """Persist this session's typed ``meta.yaml`` (disk is the source)."""
        self._meta = meta
        self.write_meta(meta)

    def materialize(self) -> None:
        """Write the session's ``meta.yaml`` (creating the dir lazily)."""
        self.write_meta(self._meta)

    @property
    def goal_summary(self) -> str:
        return self.read_session_meta().goal_summary

    @property
    def status(self) -> SessionStatusStr:
        return self.read_session_meta().status

    # ── pydantic-ai ModelMessage history (lazy codec, fs-routed) ───────────

    @property
    def messages_path(self) -> Path:
        return self.resolve() / MESSAGES_FILENAME

    def read_messages(self) -> tuple[Any, ...]:
        """Load the persisted ``ModelMessage`` tuple (``()`` if none), via fs."""
        path = self.messages_path
        if not self._fs.exists(path):
            return ()
        from molexp.agent._pydanticai.messages_codec import load_model_messages

        return load_model_messages(self._fs.read_bytes(path))

    def write_messages(self, messages: tuple[Any, ...]) -> None:
        """Persist the ``ModelMessage`` tuple via fs (empty ⇒ remove)."""
        path = self.messages_path
        if not messages:
            self._fs.remove(path)
            return
        from molexp.agent._pydanticai.messages_codec import dump_model_messages

        self._fs.write_bytes(self.messages_path, dump_model_messages(messages))


@concept_type(AGENT_KIND)
class Agent(Folder):
    """Configured agent persona — ``type = "agent.agent"``; owns sessions."""

    DEFAULT_TYPE = AGENT_KIND

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
        fs: FileSystem | None = None,
        system_prompt: str = "",
        model: str = "",
        tier: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
            fs=fs,
        )
        self._meta = AgentMeta(
            id=self._name,
            system_prompt=system_prompt,
            model=model,
            tier=tier,
            description=description,
        )

    def read_agent_meta(self) -> AgentMeta:
        """Load this agent's typed ``meta.yaml`` from disk (disk is truth)."""
        return AgentMeta.model_validate(self.read_meta().model_dump())

    def write_agent_meta(self, meta: AgentMeta) -> None:
        """Persist this agent's typed ``meta.yaml`` (disk is the source)."""
        self._meta = meta
        self.write_meta(meta)

    def materialize(self) -> None:
        """Write the agent's ``meta.yaml`` (creating the dir lazily)."""
        self.write_meta(self._meta)

    @property
    def system_prompt(self) -> str:
        return self.read_agent_meta().system_prompt

    @property
    def model(self) -> str:
        return self.read_agent_meta().model

    @property
    def tier(self) -> str:
        return self.read_agent_meta().tier

    # ── typed sugar for AgentSession children (flat concept dirs) ──────────

    def add_session(
        self, name: str, *, goal_summary: str = "", status: SessionStatusStr = "pending"
    ) -> AgentSession:
        """Create (or return) a child session Concept; idempotent on slug."""
        session = cast(AgentSession, self.add_folder(name, concept_type=AGENT_SESSION_KIND))
        if goal_summary or status != "pending":
            session.write_session_meta(
                AgentSessionMeta(id=session.name, goal_summary=goal_summary, status=status)
            )
        return session

    def get_session(self, name: str) -> AgentSession:
        return cast(AgentSession, self.get_folder(name))

    def has_session(self, name: str) -> bool:
        return self.has_folder(name)

    def list_sessions(self) -> list[AgentSession]:
        return [c for c in self.list_folders() if isinstance(c, AgentSession)]

    def remove_session(self, name: str) -> None:
        self.remove_folder(name)


__all__ = [
    "AGENT_KIND",
    "AGENT_SESSION_KIND",
    "MESSAGES_FILENAME",
    "Agent",
    "AgentSession",
]
