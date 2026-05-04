"""AgentStateStore — aggregate over per-area stores.

Bundles :class:`SessionStore`, :class:`SkillStore`, and
:class:`MemoryStore` so :class:`AgentService` can hand a single object
to runners, sessions, and routes.
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.state.memory import MemoryStore
from molexp.agent.state.sessions import SessionStore
from molexp.agent.state.skills import SkillStore


@dataclass
class AgentStateStore:
    sessions: SessionStore
    skills: SkillStore
    memory: MemoryStore
