"""AgentStateStore — aggregate over per-area stores (spec §6.4).

Bundles :class:`SessionStore`, :class:`SkillStore`, and
:class:`MemoryStore` so :class:`AgentService` can hand a single object
to runners, sessions, and routes.
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.state.memory import MemoryStore, NoopMemoryStore
from molexp.agent.state.sessions import SessionStore
from molexp.agent.state.skills import SkillStore


@dataclass
class AgentStateStore:
    sessions: SessionStore
    skills: SkillStore
    memory: MemoryStore

    @classmethod
    def in_memory(cls, sessions: SessionStore) -> "AgentStateStore":
        return cls(sessions=sessions, skills=SkillStore(), memory=NoopMemoryStore())
