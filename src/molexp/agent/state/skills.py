"""Skill: reusable behavior bundles (goal template + tool scope + system addendum).

A *skill* combines four things into one named, slash-invokable bundle:

1. A parameterized **goal template** with ``{{name}}`` placeholders
2. **System-prompt addendum** (``instructions``) applied for the session
3. **Tool scoping** via ``allowed_tools`` / ``denied_tools`` glob lists
4. A **session mode** opt-in (``default_plan_mode``, ``requires_exit_tool``)

Skills are *not* tools: they produce ``Goal`` payloads consumed by the
agent service; the agent then uses its tool surface filtered by the
skill's allow-list to fulfill the goal.

Three discovery tiers, in precedence order (later wins on
``slash_name`` collision; ``builtin`` skills are read-only and can
only be shadowed):

- **Builtin** — packaged with molexp (e.g. ``builtin-plan``).
- **User-home** — ``~/.molexp/skills.json`` (per-user, all workspaces).
- **Workspace** — ``<workspace>/.skills.json``.

All three are accessed through :class:`SkillStore`. JSON writes
(user-home + workspace tiers) are atomic temp+rename.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

SKILLS_FILE = ".skills.json"
USER_HOME_SKILLS_FILE = "skills.json"
USER_HOME_DIR_NAME = ".molexp"

_PARAM_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_SLASH_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

# Reserved slash names handled by the chat input — skills cannot reuse them.
RESERVED_SLASH_NAMES: frozenset[str] = frozenset({"plan", "clear", "model", "help"})


class SkillScope(str, Enum):
    """Where a skill lives. Higher precedence shadows lower precedence."""

    BUILTIN = "builtin"
    USER = "user"
    WORKSPACE = "workspace"


class Skill(BaseModel):
    """A saved behavior bundle (see module docstring)."""

    id: str
    name: str
    description: str = ""
    goal_template: str
    slash_name: str = ""
    instructions: str = ""
    default_plan_mode: bool = False
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    requires_exit_tool: str = ""
    builtin: bool = False
    scope: SkillScope = SkillScope.WORKSPACE
    created_at: str = ""
    updated_at: str = ""

    def required_parameters(self) -> list[str]:
        """Return the ordered list of distinct ``{{param}}`` placeholders."""

        seen: list[str] = []
        for match in _PARAM_RE.finditer(self.goal_template):
            key = match.group(1)
            if key not in seen:
                seen.append(key)
        return seen

    def materialize(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the template into a Goal-compatible dict."""

        params = params or {}

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in params:
                raise KeyError(f"Missing parameter '{key}' for skill '{self.id}'")
            return str(params[key])

        description = _PARAM_RE.sub(_sub, self.goal_template)
        return {
            "description": description,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
        }

    def matches_tool(self, tool_name: str) -> bool:
        """Glob-based allow/deny check (denial wins, empty allow ⇒ all allowed)."""

        for pattern in self.denied_tools:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return False
        if not self.allowed_tools:
            return True
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.allowed_tools)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_plan_skill() -> Skill:
    """Construct the builtin ``/plan`` skill.

    Plan mode keeps the full native tool surface; the constraint is
    on the output shape (a structured plan handoff). The instructions
    text is the canonical :data:`PLAN_MODE_ADDENDUM` so the prompt
    composer reads it directly off the skill rather than hardcoding it.
    """

    from molexp.agent.context.prompt import PLAN_MODE_ADDENDUM

    return Skill(
        id="builtin-plan",
        name="Plan mode",
        description=(
            "Plan mode. The agent explores the workspace, drafts a "
            "structured execution plan + a molexp workflow IR, and "
            "hands the bundle back for explicit approval. Tools are "
            "NOT restricted — the constraint is on the output shape, "
            "not the surface."
        ),
        goal_template="",
        slash_name="",
        instructions=PLAN_MODE_ADDENDUM,
        default_plan_mode=True,
        allowed_tools=[],
        denied_tools=[],
        requires_exit_tool="",
        builtin=True,
        scope=SkillScope.BUILTIN,
        tags=["builtin", "mode"],
    )


_BUILTIN_SKILLS: list[Skill] | None = None


def list_builtin_skills() -> list[Skill]:
    """Return every package-builtin skill, lazily constructed."""

    global _BUILTIN_SKILLS
    if _BUILTIN_SKILLS is None:
        _BUILTIN_SKILLS = [_build_plan_skill()]
    return list(_BUILTIN_SKILLS)


def get_builtin_skill(skill_id: str) -> Skill | None:
    for skill in list_builtin_skills():
        if skill.id == skill_id:
            return skill
    return None


class SkillStore:
    """Three-tier aggregating store for :class:`Skill` records."""

    def __init__(
        self,
        root: str | Path,
        user_home_dir: str | Path | None = None,
    ) -> None:
        self._workspace_path = Path(root) / SKILLS_FILE
        if user_home_dir is None:
            user_home_dir = Path.home() / USER_HOME_DIR_NAME
        self._user_path = Path(user_home_dir) / USER_HOME_SKILLS_FILE
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._workspace_path

    def path_for(self, scope: SkillScope) -> Path | None:
        if scope == SkillScope.WORKSPACE:
            return self._workspace_path
        if scope == SkillScope.USER:
            return self._user_path
        return None

    def list_all(self) -> list[Skill]:
        out: list[Skill] = []
        out.extend(list_builtin_skills())
        out.extend(self._list_disk(SkillScope.USER))
        out.extend(self._list_disk(SkillScope.WORKSPACE))
        return out

    def list_scope(self, scope: SkillScope) -> list[Skill]:
        if scope == SkillScope.BUILTIN:
            return list_builtin_skills()
        return self._list_disk(scope)

    def get(self, skill_id: str) -> Skill | None:
        for scope in (SkillScope.WORKSPACE, SkillScope.USER, SkillScope.BUILTIN):
            for skill in self.list_scope(scope):
                if skill.id == skill_id:
                    return skill
        return None

    def find_by_slash(self, slash_name: str) -> Skill | None:
        if not slash_name:
            return None
        for scope in (SkillScope.WORKSPACE, SkillScope.USER, SkillScope.BUILTIN):
            for skill in self.list_scope(scope):
                if skill.slash_name and skill.slash_name == slash_name:
                    return skill
        return None

    def create(
        self,
        name: str,
        goal_template: str,
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
        scope: SkillScope = SkillScope.WORKSPACE,
    ) -> Skill:
        if scope == SkillScope.BUILTIN:
            raise ValueError("Builtin skills are registered in code and cannot be created on disk.")
        skill = Skill(
            id=skill_id or f"skill-{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            goal_template=goal_template,
            slash_name=slash_name,
            instructions=instructions,
            default_plan_mode=default_plan_mode,
            constraints=list(constraints or []),
            success_criteria=list(success_criteria or []),
            tags=list(tags or []),
            allowed_tools=list(allowed_tools or []),
            denied_tools=list(denied_tools or []),
            requires_exit_tool=requires_exit_tool,
            builtin=False,
            scope=scope,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        with self._lock:
            data = self._read(scope)
            if any(item.get("id") == skill.id for item in data):
                raise ValueError(f"Skill '{skill.id}' already exists")
            self._validate_slash_name(skill.slash_name, exclude_id=None, data=data)
            data.append(skill.model_dump(mode="json"))
            self._write(scope, data)
        return skill

    def update(
        self,
        skill_id: str,
        scope: SkillScope = SkillScope.WORKSPACE,
        **changes: Any,
    ) -> Skill:
        if scope == SkillScope.BUILTIN:
            raise ValueError("Builtin skills are immutable.")
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
        with self._lock:
            data = self._read(scope)
            for idx, item in enumerate(data):
                if item.get("id") == skill_id:
                    if "slash_name" in changes and changes["slash_name"] is not None:
                        self._validate_slash_name(
                            str(changes["slash_name"]),
                            exclude_id=skill_id,
                            data=data,
                        )
                    item.update({k: v for k, v in changes.items() if v is not None})
                    item["updated_at"] = _now_iso()
                    data[idx] = item
                    self._write(scope, data)
                    return Skill.model_validate(item)
        raise KeyError(f"Skill '{skill_id}' not found in {scope.value} scope")

    def delete(
        self,
        skill_id: str,
        scope: SkillScope = SkillScope.WORKSPACE,
    ) -> bool:
        if scope == SkillScope.BUILTIN:
            raise ValueError(
                "Builtin skills cannot be deleted. To override one, create a "
                "same-slash skill at the workspace or user-home tier."
            )
        with self._lock:
            data = self._read(scope)
            new_data = [item for item in data if item.get("id") != skill_id]
            if len(new_data) == len(data):
                return False
            self._write(scope, new_data)
            return True

    def _list_disk(self, scope: SkillScope) -> list[Skill]:
        out: list[Skill] = []
        for item in self._read(scope):
            try:
                skill = Skill.model_validate(item)
            except Exception:
                continue
            out.append(skill.model_copy(update={"scope": scope, "builtin": False}))
        return out

    def _validate_slash_name(
        self,
        slash_name: str,
        exclude_id: str | None,
        data: list[dict[str, Any]],
    ) -> None:
        if not slash_name:
            return
        if not _SLASH_NAME_RE.match(slash_name):
            raise ValueError(
                f"Invalid slash_name '{slash_name}'. Must match [a-z0-9][a-z0-9-]{{0,31}}."
            )
        if slash_name in RESERVED_SLASH_NAMES:
            raise ValueError(f"slash_name '{slash_name}' is reserved by the chat input.")
        for item in data:
            if exclude_id is not None and item.get("id") == exclude_id:
                continue
            if item.get("slash_name") == slash_name:
                raise ValueError(
                    f"slash_name '{slash_name}' is already used by skill '{item.get('id')}'."
                )

    def _read(self, scope: SkillScope) -> list[dict[str, Any]]:
        path = self.path_for(scope)
        if path is None or not path.exists():
            return []
        try:
            content = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(content, list):
            return [item for item in content if isinstance(item, dict)]
        return []

    def _write(self, scope: SkillScope, data: list[dict[str, Any]]) -> None:
        path = self.path_for(scope)
        if path is None:
            raise ValueError(f"Cannot write to scope {scope.value}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, path)


__all__ = [
    "RESERVED_SLASH_NAMES",
    "SKILLS_FILE",
    "Skill",
    "SkillScope",
    "SkillStore",
    "USER_HOME_DIR_NAME",
    "USER_HOME_SKILLS_FILE",
    "get_builtin_skill",
    "list_builtin_skills",
]
