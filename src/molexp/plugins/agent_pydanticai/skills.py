"""Skill: reusable behavior bundles (goal template + tool scope + system addendum).

A *skill* combines four things into one named, slash-invokable bundle:

1. A parameterized **goal template** with ``{{name}}`` placeholders
2. **System-prompt addendum** (``instructions``) applied for the session
3. **Tool scoping** via ``allowed_tools`` / ``denied_tools`` glob lists
4. A **session mode** opt-in (``default_plan_mode``, ``requires_exit_tool``)

It is **not** a tool: skills produce ``Goal`` payloads consumed by
``AgentService.run``; the agent then uses its tool surface (native + MCP)
filtered down to the skill's allow-list to fulfill the goal.

Three discovery tiers, listed in order of precedence (later wins on
``slash_name`` collision, but ``builtin`` skills can never be deleted —
they can only be **shadowed** by a same-slash entry from a higher tier):

- **Builtin** — packaged with molexp, registered in
  :mod:`._pydantic_ai.builtin_skills`
- **User-home** — ``~/.molexp/skills.json`` (per-user, all workspaces)
- **Workspace** — ``<workspace>/.skills.json``

All three tiers are accessed through the same :class:`SkillStore` API.
JSON writes (user-home + workspace tiers) are atomic (temp file + rename).
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
# ``/plan`` stays here because the chat client owns the next-message-toggle
# UX for it; the matching :data:`PLAN_SKILL` builtin is looked up by ID at
# runtime, not by slash.
RESERVED_SLASH_NAMES: frozenset[str] = frozenset({"plan", "clear", "model", "help"})


class SkillScope(str, Enum):
    """Where a skill lives. Higher precedence shadows lower precedence."""

    BUILTIN = "builtin"
    USER = "user"
    WORKSPACE = "workspace"


class Skill(BaseModel):
    """A saved behavior bundle.

    The ``goal_template`` may contain ``{{name}}`` placeholders that are
    substituted from a parameters dict at instantiation time. Unknown
    placeholders raise ``KeyError`` from :meth:`materialize`.

    A skill with a non-empty ``slash_name`` can also be invoked from the
    chat input as ``/<slash_name> key=value …``; otherwise it remains a
    launcher-only template (driven from the Settings UI).

    ``instructions`` are appended to the system prompt when the skill is
    used to start a session. ``default_plan_mode`` flips the new session
    into read-only plan mode by default; the user can override.

    ``allowed_tools`` / ``denied_tools`` are fnmatch-style glob lists
    applied against tool names (e.g. ``"list_*"``, ``"mcp:python.*"``).
    Empty allow-list = "every tool that isn't denied". Non-empty
    allow-list narrows the surface to matches only. Denial wins ties.

    ``requires_exit_tool``, when set, names a builtin tool the agent
    MUST call to exit the skill mode (used by ``/plan`` to force a
    structured plan handoff via ``exit_plan_mode``).

    ``scope`` is set by :class:`SkillStore` at load time and reflects
    which tier owns this record. ``builtin=True`` means this entry was
    registered in code, not stored on disk.
    """

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
        """Return the ordered list of distinct ``{{param}}`` placeholders.

        Used by the chat parser to validate that ``key=value`` arguments
        cover every required slot before launching a session.
        """
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
        """Return True if a tool with this name is exposed by this skill.

        Glob semantics:

        - Denial wins: any matching pattern in ``denied_tools`` excludes
          the tool, regardless of ``allowed_tools``.
        - Empty allow-list = allow-everything-not-denied.
        - Non-empty allow-list = expose only matching tools.

        Patterns use :mod:`fnmatch` (``*``, ``?``, ``[abc]``). MCP tool
        names are typically prefixed by their server (``server.tool``),
        so ``"mcp:python.*"`` matches every tool from the ``python`` MCP.
        """
        for pattern in self.denied_tools:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return False
        if not self.allowed_tools:
            return True
        return any(
            fnmatch.fnmatchcase(tool_name, pattern)
            for pattern in self.allowed_tools
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillStore:
    """Three-tier aggregating store for :class:`Skill` records.

    Tiers (high precedence → low):

    1. **Workspace** — ``<workspace>/.skills.json``
    2. **User-home** — ``~/.molexp/skills.json`` (override via
       ``user_home_dir`` ctor arg, primarily for tests)
    3. **Builtin** — registered in code via
       :mod:`.builtin_skills`; cannot be deleted, only shadowed

    "Shadow" semantics: when two tiers contain a skill with the same
    ``slash_name``, the higher-precedence tier wins for invocation
    (``find_by_slash``). :meth:`list_all` returns the union with
    ``scope`` correctly populated, in display order builtin → user →
    workspace; shadowed entries carry their original tier (callers can
    detect shadowing by grouping on ``slash_name``).

    Reads parse the JSON files lazily; writes target a single tier
    (workspace by default) and are atomic + serialized via an internal
    lock so concurrent route handlers in one process cannot interleave.
    """

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
        """Workspace-tier path. Kept for backward compatibility."""
        return self._workspace_path

    def path_for(self, scope: SkillScope) -> Path | None:
        """Return the on-disk path backing ``scope``, or None for builtin."""
        if scope == SkillScope.WORKSPACE:
            return self._workspace_path
        if scope == SkillScope.USER:
            return self._user_path
        return None

    # ── Reads ────────────────────────────────────────────────────────────

    def list_all(self) -> list[Skill]:
        """Return every skill across all tiers, in display order.

        Display order: builtin → user → workspace. Shadowed entries are
        included with their original ``scope`` field so the UI can render
        the override relationship explicitly. Use :meth:`find_by_slash`
        when you want a single resolved skill for invocation.
        """
        out: list[Skill] = []
        for skill in self._list_builtin():
            out.append(skill)
        for skill in self._list_disk(SkillScope.USER):
            out.append(skill)
        for skill in self._list_disk(SkillScope.WORKSPACE):
            out.append(skill)
        return out

    def list_scope(self, scope: SkillScope) -> list[Skill]:
        """Return only skills belonging to ``scope`` (no aggregation)."""
        if scope == SkillScope.BUILTIN:
            return self._list_builtin()
        return self._list_disk(scope)

    def get(self, skill_id: str) -> Skill | None:
        """Look up a skill by ID across all tiers (workspace > user > builtin)."""
        for scope in (SkillScope.WORKSPACE, SkillScope.USER, SkillScope.BUILTIN):
            for skill in self.list_scope(scope):
                if skill.id == skill_id:
                    return skill
        return None

    def find_by_slash(self, slash_name: str) -> Skill | None:
        """Resolve ``/<slash_name>`` to a single skill via shadow precedence.

        Workspace overrides user-home overrides builtin. Empty
        ``slash_name`` returns None (those skills are launcher-only).
        """
        if not slash_name:
            return None
        for scope in (SkillScope.WORKSPACE, SkillScope.USER, SkillScope.BUILTIN):
            for skill in self.list_scope(scope):
                if skill.slash_name and skill.slash_name == slash_name:
                    return skill
        return None

    # ── Writes (workspace + user-home only; builtin is read-only) ────────

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

    # ── Internals ────────────────────────────────────────────────────────

    def _list_builtin(self) -> list[Skill]:
        from .builtin_skills import list_builtin_skills

        return list_builtin_skills()

    def _list_disk(self, scope: SkillScope) -> list[Skill]:
        out: list[Skill] = []
        for item in self._read(scope):
            try:
                skill = Skill.model_validate(item)
            except Exception:
                continue
            # Force-tag with the tier we read from — the on-disk JSON may
            # be stale or hand-edited and we trust the source path more
            # than whatever the file claims.
            out.append(skill.model_copy(update={"scope": scope, "builtin": False}))
        return out

    def _validate_slash_name(
        self,
        slash_name: str,
        exclude_id: str | None,
        data: list[dict[str, Any]],
    ) -> None:
        """Validate format, reservation, and per-tier uniqueness of ``slash_name``.

        Empty string is allowed (launcher-only skill). Otherwise the value
        must match ``[a-z0-9][a-z0-9-]{0,31}``, must not collide with the
        reserved chat builtins, and must be unique within the same tier.
        Cross-tier collisions are permitted on purpose — that is the
        shadow mechanism users may rely on to override a builtin.
        """
        if not slash_name:
            return
        if not _SLASH_NAME_RE.match(slash_name):
            raise ValueError(
                f"Invalid slash_name '{slash_name}'. Must match "
                "[a-z0-9][a-z0-9-]{0,31}."
            )
        if slash_name in RESERVED_SLASH_NAMES:
            raise ValueError(
                f"slash_name '{slash_name}' is reserved by the chat input."
            )
        for item in data:
            if exclude_id is not None and item.get("id") == exclude_id:
                continue
            if item.get("slash_name") == slash_name:
                raise ValueError(
                    f"slash_name '{slash_name}' is already used by skill "
                    f"'{item.get('id')}'."
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
