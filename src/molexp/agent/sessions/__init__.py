"""Session persistence subsystem — metadata + JSONL streams + catalog.

Legacy flat-layout session storage at ``<workspace_root>/sessions/<sid>/``.
Retained for backward compatibility with the ``AgentRunner`` chat-mode
persistence path; new code should use :class:`molexp.agent.folders.Agent`
+ :class:`molexp.agent.folders.AgentSession` (Folder subclasses) which
mount sessions under their owning Agent at
``<workspace_root>/agents/<aid>/agent_sessions/<sid>/``.
"""

from molexp.agent.sessions.catalog import (
    MODEL_MESSAGES_FILENAME,
    SESSION_METADATA_FILENAME,
    SESSIONS_DIRNAME,
    SessionCatalog,
)
from molexp.agent.sessions.store import SessionStore
from molexp.agent.sessions.types import SessionMetadata

__all__ = [
    "MODEL_MESSAGES_FILENAME",
    "SESSIONS_DIRNAME",
    "SESSION_METADATA_FILENAME",
    "SessionCatalog",
    "SessionMetadata",
    "SessionStore",
]
