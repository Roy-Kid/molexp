"""Agent-layer Concept types — ``Agent`` + ``AgentSession``.

Rehomed onto :class:`molexp.workspace.Folder` (the OKF rewrite, wsokf-06): an
``Agent`` is a workspace Concept (``kind = "agent.agent"``) whose
``AgentSession`` children (``kind = "agent.session"``) are **flat** child
Concepts — one dir per session, ``meta.yaml`` for structured identity,
``messages.jsonl`` for the pydantic-ai history. Both register with the shared
concept-type registry (``molexp.knowledge.types.concept_type`` — the *only*
knowledge edge), so ``workspace.folder.concept_from_dir`` / ``list_folders`` /
``get_folder`` rebuild the right subclass. All I/O routes through the Folder's
injectable filesystem (``self._fs``), so a session works against any backend.

Unlike the base workspace ``Folder`` (whose ``meta.yaml`` is an additive
``{type, id}`` marker alongside an authoritative ``metadata.json``), the agent
Concepts have **no** separate entity json — their settled identity
(system_prompt/model/tier for Agent; goal_summary/status/timestamps for
AgentSession) *is* the OKF structured identity, so ``meta.yaml`` is the full,
rich authority. ``Agent`` / ``AgentSession`` therefore override
``write_meta`` / ``materialize`` / ``from_disk`` to make the typed
:class:`~molexp.agent.folders_metadata.AgentMeta` /
:class:`~molexp.agent.folders_metadata.AgentSessionMeta` the meta.yaml payload.

The *runtime* ``AgentSession`` (in :mod:`molexp.agent.session`) is a distinct
in-memory class; this one is its on-disk persistent counterpart.
"""

from __future__ import annotations

import yaml

from molexp._typing import JSONValue
from molexp.agent.folders_metadata import AgentMeta, AgentSessionMeta, SessionStatusStr
from molexp.knowledge.types import concept_type
from molexp.path import Path
from molexp.workspace import Folder
from molexp.workspace.fs import FileSystem, PathArg
from molexp.workspace.models import FolderMetadata

AGENT_KIND = "agent.agent"
AGENT_SESSION_KIND = "agent.session"
MESSAGES_FILENAME = "messages.jsonl"
META_YAML_FILENAME = "meta.yaml"


def _folder_metadata(slug: str, kind: str) -> FolderMetadata:
    """Build the base :class:`FolderMetadata` an agent Concept carries."""
    return FolderMetadata(id=slug, name=slug, kind=kind)


@concept_type(AGENT_SESSION_KIND)
class AgentSession(Folder):
    """One conversation under an :class:`Agent` — ``kind = "agent.session"``.

    Holds ``meta.yaml`` (:class:`AgentSessionMeta`, the rich identity authority)
    + ``messages.jsonl`` (the pydantic-ai history, written via
    :meth:`write_messages` through ``self._fs``).
    """

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root_path: PathArg | None = None,
        kind: str = AGENT_SESSION_KIND,
        fs: FileSystem | None = None,
        goal_summary: str = "",
        status: SessionStatusStr = "pending",
        _meta: AgentSessionMeta | None = None,
    ) -> None:
        super().__init__(name=name, parent=parent, kind=kind, root_path=root_path, fs=fs)
        self._session_meta = _meta or AgentSessionMeta(
            id=self._name, goal_summary=goal_summary, status=status
        )

    # ── meta.yaml authority (rich AgentSessionMeta, not the additive marker) ─

    def write_meta(self) -> str:
        """Write the rich :class:`AgentSessionMeta` as this session's meta.yaml."""
        fpath = self._fs.join(self.path(), META_YAML_FILENAME)
        self._fs.atomic_write_text(
            fpath, yaml.safe_dump(self._session_meta.model_dump(mode="json"), sort_keys=False)
        )
        return fpath

    def read_meta(self) -> dict[str, JSONValue]:
        """Read the raw meta.yaml dict, or ``{}`` if absent."""
        fpath = self._fs.join(self.resolve(), META_YAML_FILENAME)
        if not self._fs.exists(fpath):
            return {}
        loaded = yaml.safe_load(self._fs.read_text(fpath)) or {}
        return loaded if isinstance(loaded, dict) else {}

    def read_session_meta(self) -> AgentSessionMeta:
        """Load this session's typed ``meta.yaml`` from disk (disk is truth)."""
        return AgentSessionMeta.model_validate(self.read_meta())

    def write_session_meta(self, meta: AgentSessionMeta) -> None:
        """Persist this session's typed ``meta.yaml`` (disk is the source)."""
        self._session_meta = meta
        self.write_meta()

    def materialize(self) -> None:
        """Write the session's ``meta.yaml`` (creating the dir lazily)."""
        self.write_meta()

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> AgentSession:
        """Reconstruct an :class:`AgentSession` from its ``meta.yaml`` (disk is truth)."""
        fs = parent._fs
        meta_path = fs.join(child_dir, META_YAML_FILENAME)
        slug = fs.basename(child_dir)
        meta = (
            AgentSessionMeta.model_validate(yaml.safe_load(fs.read_text(meta_path)) or {})
            if fs.exists(meta_path)
            else AgentSessionMeta(id=slug)
        )
        session = cls(name=slug, parent=parent, _meta=meta)
        session._metadata = _folder_metadata(slug, AGENT_SESSION_KIND)
        return session

    @property
    def goal_summary(self) -> str:
        return self.read_session_meta().goal_summary

    @property
    def status(self) -> SessionStatusStr:
        return self.read_session_meta().status

    # ── pydantic-ai ModelMessage history (lazy codec, fs-routed) ───────────

    @property
    def messages_path(self) -> Path:
        return Path(self._fs.join(self.resolve(), MESSAGES_FILENAME))

    def read_messages(self) -> tuple[object, ...]:
        """Load the persisted ``ModelMessage`` tuple (``()`` if none), via fs."""
        path = self.messages_path
        if not self._fs.exists(path):
            return ()
        from molexp.agent._pydanticai.messages_codec import load_model_messages

        return load_model_messages(self._fs.read_bytes(path))

    def write_messages(self, messages: tuple[object, ...]) -> None:
        """Persist the ``ModelMessage`` tuple via fs (empty ⇒ remove)."""
        path = self.messages_path
        if not messages:
            self._fs.remove(path)
            return
        from molexp.agent._pydanticai.messages_codec import dump_model_messages

        self._fs.write_bytes(path, dump_model_messages(messages))


@concept_type(AGENT_KIND)
class Agent(Folder):
    """Configured agent persona — ``kind = "agent.agent"``; owns sessions."""

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root_path: PathArg | None = None,
        kind: str = AGENT_KIND,
        fs: FileSystem | None = None,
        system_prompt: str = "",
        model: str = "",
        tier: str = "",
        description: str = "",
        _meta: AgentMeta | None = None,
    ) -> None:
        super().__init__(name=name, parent=parent, kind=kind, root_path=root_path, fs=fs)
        self._agent_meta = _meta or AgentMeta(
            id=self._name,
            system_prompt=system_prompt,
            model=model,
            tier=tier,
            description=description,
        )

    # ── meta.yaml authority (rich AgentMeta, not the additive marker) ──────

    def write_meta(self) -> str:
        """Write the rich :class:`AgentMeta` as this agent's meta.yaml."""
        fpath = self._fs.join(self.path(), META_YAML_FILENAME)
        self._fs.atomic_write_text(
            fpath, yaml.safe_dump(self._agent_meta.model_dump(mode="json"), sort_keys=False)
        )
        return fpath

    def read_meta(self) -> dict[str, JSONValue]:
        """Read the raw meta.yaml dict, or ``{}`` if absent."""
        fpath = self._fs.join(self.resolve(), META_YAML_FILENAME)
        if not self._fs.exists(fpath):
            return {}
        loaded = yaml.safe_load(self._fs.read_text(fpath)) or {}
        return loaded if isinstance(loaded, dict) else {}

    def read_agent_meta(self) -> AgentMeta:
        """Load this agent's typed ``meta.yaml`` from disk (disk is truth)."""
        return AgentMeta.model_validate(self.read_meta())

    def write_agent_meta(self, meta: AgentMeta) -> None:
        """Persist this agent's typed ``meta.yaml`` (disk is the source)."""
        self._agent_meta = meta
        self.write_meta()

    def materialize(self) -> None:
        """Write the agent's ``meta.yaml`` (creating the dir lazily)."""
        self.write_meta()

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Agent:
        """Reconstruct an :class:`Agent` from its ``meta.yaml`` (disk is truth)."""
        fs = parent._fs
        meta_path = fs.join(child_dir, META_YAML_FILENAME)
        slug = fs.basename(child_dir)
        meta = (
            AgentMeta.model_validate(yaml.safe_load(fs.read_text(meta_path)) or {})
            if fs.exists(meta_path)
            else AgentMeta(id=slug)
        )
        agent = cls(name=slug, parent=parent, _meta=meta)
        agent._metadata = _folder_metadata(slug, AGENT_KIND)
        return agent

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
        child = AgentSession(parent=self, name=name, goal_summary=goal_summary, status=status)
        session = self.add_folder(child)
        assert isinstance(session, AgentSession)
        return session

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
    "AGENT_SESSION_KIND",
    "MESSAGES_FILENAME",
    "Agent",
    "AgentSession",
]
