"""Three-tier store for :class:`Skill` records.

Backed by :class:`~molexp.agent.persistence.TieredResourceStore`:

- **native** — registered in code via :meth:`SkillStore.register`
- **user** — ``~/.molexp/skills.json``
- **workspace** — ``<root>/.skills.json``

Workspace shadows user shadows native on collision by ``id``. Adds a
small skill-specific layer on top of the generic store: id generation,
slash-name validation, slash-name uniqueness across non-shadowed
entries.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, ClassVar

from molexp.agent.persistence import Scope, TieredResourceStore
from molexp.agent.skills.types import (
    RESERVED_SLASH_NAMES,
    SLASH_NAME_RE,
    Skill,
)

SKILLS_FILE = ".skills.json"
USER_HOME_DIR_NAME = ".molexp"
USER_HOME_SKILLS_FILE = "skills.json"
KIND_KEY = "skills"


class SkillStore(TieredResourceStore[Skill]):
    """Workspace-scoped tiered store of saved skills."""

    _registrations: ClassVar[list[Skill]] = []

    def __init__(
        self,
        root: str | Path,
        user_home_dir: str | Path | None = None,
    ) -> None:
        if user_home_dir is None:
            user_home_dir = Path.home() / USER_HOME_DIR_NAME
        super().__init__(
            user_path=Path(user_home_dir) / USER_HOME_SKILLS_FILE,
            workspace_path=Path(root) / SKILLS_FILE,
            spec_cls=Skill,
            kind_key=KIND_KEY,
        )

    @property
    def path(self) -> Path:
        return self._workspace_path

    def find_by_slash(self, slash_name: str) -> Skill | None:
        """Return the highest-priority unshadowed skill bound to ``slash_name``."""

        if not slash_name:
            return None
        for entry in self.list_all():
            if entry.shadowed:
                continue
            if entry.slash_name and entry.slash_name == slash_name:
                return entry
        return None

    def create(  # type: ignore[override]
        self,
        scope: Scope = Scope.WORKSPACE,
        *,
        name: str,
        goal_template: str = "",
        description: str = "",
        slash_name: str = "",
        instructions: str = "",
        default_plan_mode: bool = False,
        constraints: list[str] | None = None,
        success_criteria: list[str] | None = None,
        tags: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        requires_exit_tool: str = "",
        skill_id: str | None = None,
    ) -> Skill:
        if scope is Scope.NATIVE:
            raise ValueError("Native skills are registered in code; use SkillStore.register().")
        self._validate_slash_name(slash_name, exclude_id=None)
        return super().create(
            scope,
            id=skill_id or f"skill-{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            tags=list(tags or []),
            goal_template=goal_template,
            slash_name=slash_name,
            instructions=instructions,
            default_plan_mode=default_plan_mode,
            constraints=list(constraints or []),
            success_criteria=list(success_criteria or []),
            allowed_tools=list(allowed_tools or []),
            denied_tools=list(denied_tools or []),
            requires_exit_tool=requires_exit_tool,
        )

    def update(  # type: ignore[override]
        self,
        skill_id: str,
        scope: Scope = Scope.WORKSPACE,
        **changes: Any,
    ) -> Skill:
        if scope is Scope.NATIVE:
            raise ValueError("Native skills are immutable.")
        allowed = {
            "name",
            "description",
            "goal_template",
            "slash_name",
            "instructions",
            "default_plan_mode",
            "constraints",
            "success_criteria",
            "tags",
            "allowed_tools",
            "denied_tools",
            "requires_exit_tool",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown skill fields: {sorted(unknown)}")
        new_slash = changes.get("slash_name")
        if new_slash:
            self._validate_slash_name(str(new_slash), exclude_id=skill_id)
        return super().update(
            skill_id, scope, **{k: v for k, v in changes.items() if v is not None}
        )

    def delete(  # type: ignore[override]
        self,
        skill_id: str,
        scope: Scope = Scope.WORKSPACE,
    ) -> bool:
        if scope is Scope.NATIVE:
            raise ValueError(
                "Native skills cannot be deleted. To override one, "
                "create a same-slash skill at the workspace or user tier."
            )
        return super().delete(skill_id, scope)

    def _validate_slash_name(self, slash_name: str, exclude_id: str | None) -> None:
        if not slash_name:
            return
        if not SLASH_NAME_RE.match(slash_name):
            raise ValueError(
                f"Invalid slash_name '{slash_name}'. Must match [a-z0-9][a-z0-9-]{{0,31}}."
            )
        if slash_name in RESERVED_SLASH_NAMES:
            raise ValueError(f"slash_name '{slash_name}' is reserved by the chat input.")
        for entry in self.list_all():
            if entry.shadowed:
                continue
            if exclude_id is not None and entry.id == exclude_id:
                continue
            if entry.slash_name == slash_name:
                raise ValueError(
                    f"slash_name '{slash_name}' is already used by skill '{entry.id}'."
                )


__all__ = [
    "KIND_KEY",
    "SKILLS_FILE",
    "SkillStore",
    "USER_HOME_DIR_NAME",
    "USER_HOME_SKILLS_FILE",
]
