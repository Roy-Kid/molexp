"""Agent-layer :class:`Folder` subclasses — ``Agent`` + ``AgentSession``.

Per sub-spec ``unify-folder-abstraction-03`` § Design § 4, the agent
layer owns its on-disk entity tree by subclassing the public
:class:`molexp.workspace.Folder`. workspace remains unaware of these
classes; the kinds ``agent.agent`` / ``agent.session`` and the
metadata shapes are agent-layer-internal.

Mount points are the caller's choice — any workspace ``Folder`` accepts
``Agent`` (or any sibling) via the generic ``add_folder(...)`` API::

    ws = Workspace("./lab")
    proj = ws.add_project("qm9")

    # Pattern A — workspace-level agent (shared across projects)
    review_bot = ws.add_folder(Agent(name="review-bot"))
    sess = review_bot.add_session("chat-1")

    # Pattern B — project-scoped agent
    helper = proj.add_folder(Agent(name="qm9-helper"))

``AgentSession.read_messages`` / ``write_messages`` lazy-load the
pydantic-ai message codec so ``import molexp.agent.folders`` does **not**
pull ``pydantic_ai`` into ``sys.modules``; the codec is invoked only
when a caller actually reads/writes the conversation history.

Naming note: the *runtime* ``AgentSession`` (in
:mod:`molexp.agent.session`) is a transient in-memory object passed to
``AgentRunner.run``; this :class:`AgentSession` is its on-disk
persistent counterpart (subclasses :class:`Folder`). They are distinct
classes living in distinct modules — ``from molexp.agent.folders import
AgentSession`` for the storage class, ``from molexp.agent.session
import AgentSession`` (or ``from molexp.agent import AgentSession``)
for the runtime class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from molexp._typing import JSONValue
from molexp.agent.folders_metadata import (
    AgentMetadata,
    AgentSessionMetadata,
    SessionStatusStr,
)
from molexp.workspace import Folder, FolderMetadata
from molexp.workspace.base import (
    _load_metadata,
    _reconstruct,
    _save_metadata,
)

AGENT_KIND = "agent.agent"
AGENT_SESSION_KIND = "agent.session"

AGENT_METADATA_FILENAME = "agent.json"
AGENT_SESSION_METADATA_FILENAME = "agent_session.json"
MESSAGES_FILENAME = "messages.jsonl"


# ── AgentSession (Folder subclass) ─────────────────────────────────────────


class AgentSession(Folder):
    """One conversation under an :class:`Agent` — ``kind = "agent.session"``.

    Per-session dir holds:

    - ``agent_session.json`` — :class:`AgentSessionMetadata` payload
      (goal_summary / status / timestamps).
    - ``messages.jsonl`` — pydantic-ai ``ModelMessage`` history,
      written via :meth:`write_messages` (lazy codec).
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = AGENT_SESSION_KIND,
        goal_summary: str = "",
        status: SessionStatusStr = "pending",
        _entity_metadata: AgentSessionMetadata | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else AgentSessionMetadata(
                id=self._name,
                name=name,
                kind=kind,
                goal_summary=goal_summary,
                status=status,
            )
        )
        self._entity_metadata: AgentSessionMetadata = meta

    def _compute_path(self) -> Path:
        """Keep this in agreement with :meth:`_child_dir`."""
        if self._parent is None:
            raise RuntimeError(
                f"AgentSession {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self)._child_dir(self._parent, self._name)

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Sessions live under ``<parent>/agent_sessions/<id>/``."""
        return parent.path() / "agent_sessions" / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> AgentSession:
        meta = _load_metadata(AgentSessionMetadata, child_dir / AGENT_SESSION_METADATA_FILENAME)
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": AGENT_SESSION_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.name,
                    kind=AGENT_SESSION_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.updated_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
            },
        )

    @property
    def metadata(self) -> AgentSessionMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def goal_summary(self) -> str:
        return self._entity_metadata.goal_summary

    @property
    def status(self) -> SessionStatusStr:
        return self._entity_metadata.status

    def materialize(self) -> None:
        self.path().mkdir(parents=True, exist_ok=True)
        _save_metadata(self._entity_metadata, self.path() / AGENT_SESSION_METADATA_FILENAME)

    def save(self) -> None:
        _save_metadata(self._entity_metadata, self.path() / AGENT_SESSION_METADATA_FILENAME)

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    # ── pydantic-ai ModelMessage history (lazy codec) ──────────────────────

    @property
    def messages_path(self) -> Path:
        return self.path() / MESSAGES_FILENAME

    def read_messages(self) -> tuple[Any, ...]:
        """Load the persisted pydantic-ai ``ModelMessage`` tuple.

        Returns the empty tuple when no history file exists. The codec
        lives behind the ``_pydanticai/`` import-boundary firewall, so
        importing :mod:`molexp.agent.folders` does NOT pull
        ``pydantic_ai`` in — only the first call to
        ``read_messages``/``write_messages`` does.
        """
        path = self.messages_path
        if not path.exists():
            return ()
        # function-local import — pydantic-ai stays lazy
        from molexp.agent._pydanticai.messages_codec import load_model_messages

        return load_model_messages(path.read_bytes())

    def write_messages(self, messages: tuple[Any, ...]) -> None:
        """Atomically persist the pydantic-ai ``ModelMessage`` tuple."""
        path = self.messages_path
        if not messages:
            if path.exists():
                path.unlink()
            return
        from molexp.agent._pydanticai.messages_codec import dump_model_messages

        self.path().mkdir(parents=True, exist_ok=True)
        payload = dump_model_messages(messages)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)


# ── Agent (Folder subclass) ────────────────────────────────────────────────


class Agent(Folder):
    """Configured agent persona — ``kind = "agent.agent"``.

    Owns multiple :class:`AgentSession` children via typed semantic-sugar
    CRUD (``add_session / get_session / has_session / list_sessions /
    remove_session``), all one-line wrappers over the generic
    :class:`Folder` CRUD.
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = AGENT_KIND,
        system_prompt: str = "",
        model: str = "",
        tier: str = "",
        description: str = "",
        _entity_metadata: AgentMetadata | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else AgentMetadata(
                id=self._name,
                name=name,
                kind=kind,
                system_prompt=system_prompt,
                model=model,
                tier=tier,
                description=description,
            )
        )
        self._entity_metadata: AgentMetadata = meta

    def _compute_path(self) -> Path:
        """Keep this in agreement with :meth:`_child_dir`."""
        if self._parent is None:
            raise RuntimeError(
                f"Agent {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self)._child_dir(self._parent, self._name)

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Agents live under ``<parent>/agents/<id>/``."""
        return parent.path() / "agents" / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> Agent:
        meta = _load_metadata(AgentMetadata, child_dir / AGENT_METADATA_FILENAME)
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": AGENT_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.name,
                    kind=AGENT_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.updated_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
            },
        )

    @property
    def metadata(self) -> AgentMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def system_prompt(self) -> str:
        return self._entity_metadata.system_prompt

    @property
    def model(self) -> str:
        return self._entity_metadata.model

    @property
    def tier(self) -> str:
        return self._entity_metadata.tier

    def materialize(self) -> None:
        self.path().mkdir(parents=True, exist_ok=True)
        _save_metadata(self._entity_metadata, self.path() / AGENT_METADATA_FILENAME)

    def save(self) -> None:
        _save_metadata(self._entity_metadata, self.path() / AGENT_METADATA_FILENAME)

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    # ── Typed semantic-sugar CRUD for AgentSession children ────────────────

    def add_session(self, name: str, **kwargs: Any) -> AgentSession:
        return cast(AgentSession, self.add_folder(AgentSession(name=name, **kwargs)))

    def get_session(self, name: str) -> AgentSession:
        return self.get_folder(name, cls=AgentSession)

    def has_session(self, name: str) -> bool:
        return self.has_folder(name, cls=AgentSession)

    def list_sessions(self) -> list[AgentSession]:
        return self.list_folders(cls=AgentSession)

    def remove_session(self, name: str) -> None:
        self.remove_folder(name, cls=AgentSession)


__all__ = [
    "AGENT_KIND",
    "AGENT_METADATA_FILENAME",
    "AGENT_SESSION_KIND",
    "AGENT_SESSION_METADATA_FILENAME",
    "MESSAGES_FILENAME",
    "Agent",
    "AgentSession",
]
