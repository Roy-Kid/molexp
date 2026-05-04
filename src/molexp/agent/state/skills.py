"""Skill store (spec §6.4).

Skills are slash-commands with a markdown body. The store is the
single source of truth so :class:`ContextManager` can re-resolve the
addendum every turn (spec §5.1).

Phase 1a only needs the in-memory store; Phase 2 wires up workspace
JSON persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    """User- or workspace-defined slash command."""

    id: str
    name: str
    body: str
    scope: str = "workspace"
    tools: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class SkillStore:
    """In-memory skill registry."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills = {s.id: s for s in skills}

    def upsert(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def remove(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def list(self) -> tuple[Skill, ...]:
        return tuple(sorted(self._skills.values(), key=lambda s: s.id))

    def addendum(self, skill_id: str | None) -> str:
        if skill_id is None:
            return ""
        skill = self.get(skill_id)
        return skill.body if skill else ""
