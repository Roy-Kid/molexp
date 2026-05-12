"""Session persistence subsystem — metadata + JSONL streams + catalog.

The :class:`SessionCatalog` (formerly ``workspace.sessions.SessionLibrary``)
combines per-session ``session.json`` files with a flat row index, both
stored under ``<workspace_root>/.subsystems/agent.sessions/`` via
workspace's :class:`SubsystemStore`.
"""

from molexp.agent.sessions.catalog import (
    MODEL_MESSAGES_FILENAME,
    SESSION_METADATA_FILENAME,
    SESSIONS_SUBSYSTEM_KIND,
    SessionCatalog,
)
from molexp.agent.sessions.store import SessionStore
from molexp.agent.sessions.types import SessionMetadata

__all__ = [
    "MODEL_MESSAGES_FILENAME",
    "SESSIONS_SUBSYSTEM_KIND",
    "SESSION_METADATA_FILENAME",
    "SessionCatalog",
    "SessionMetadata",
    "SessionStore",
]
