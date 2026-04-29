"""Skill: reusable goal templates persisted in the workspace.

A *skill* is a named, parameterized goal — e.g. "plot ENERGY vs TEMP in
project X" — that the user can save once and re-launch with one click.
It is **not** a tool: skills produce ``Goal`` payloads consumed by
``AgentService.run``; the agent then uses its existing tool surface
(native + MCP) to fulfill the goal.

The store is workspace-scoped: each workspace owns a ``.skills.json``
file that is written atomically (temp file + ``os.rename``).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

SKILLS_FILE = ".skills.json"

_PARAM_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_SLASH_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

# Reserved slash names handled by the chat input — skills cannot reuse them.
RESERVED_SLASH_NAMES: frozenset[str] = frozenset({"plan", "clear", "model", "help"})


class Skill(BaseModel):
    """A saved goal template.

    The ``goal_template`` may contain ``{{name}}`` placeholders that are
    substituted from a parameters dict at instantiation time. Unknown
    placeholders raise ``KeyError`` from :meth:`materialize`.

    A skill with a non-empty ``slash_name`` can also be invoked from the
    chat input as ``/<slash_name> key=value …``; otherwise it remains a
    launcher-only template (driven from the Settings UI).

    ``instructions`` are appended to the system prompt when the skill is
    used to start a session. ``default_plan_mode`` flips the new session
    into read-only plan mode by default; the user can override.
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillStore:
    """File-backed store for :class:`Skill` records.

    Reads parse the JSON file lazily; writes are atomic and serialized
    by an internal lock so concurrent route handlers within one
    process cannot interleave updates.
    """

    def __init__(self, root: str | Path) -> None:
        self._path = Path(root) / SKILLS_FILE
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def list_all(self) -> list[Skill]:
        data = self._read()
        return [Skill.model_validate(item) for item in data]

    def get(self, skill_id: str) -> Skill | None:
        for skill in self.list_all():
            if skill.id == skill_id:
                return skill
        return None

    def find_by_slash(self, slash_name: str) -> Skill | None:
        """Return the skill bound to ``/<slash_name>`` or ``None``.

        Lookup is case-sensitive and skips skills with empty ``slash_name``
        — those are launcher-only.
        """
        if not slash_name:
            return None
        for skill in self.list_all():
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
        skill_id: str | None = None,
    ) -> Skill:
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
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        with self._lock:
            data = self._read()
            if any(item.get("id") == skill.id for item in data):
                raise ValueError(f"Skill '{skill.id}' already exists")
            self._validate_slash_name(skill.slash_name, exclude_id=None, data=data)
            data.append(skill.model_dump())
            self._write(data)
        return skill

    def update(self, skill_id: str, **changes: Any) -> Skill:
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
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown skill fields: {sorted(unknown)}")
        with self._lock:
            data = self._read()
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
                    self._write(data)
                    return Skill.model_validate(item)
        raise KeyError(f"Skill '{skill_id}' not found")

    def _validate_slash_name(
        self,
        slash_name: str,
        exclude_id: str | None,
        data: list[dict[str, Any]],
    ) -> None:
        """Validate format, reservation, and uniqueness of ``slash_name``.

        Empty string is allowed (launcher-only skill). Otherwise the value
        must match ``[a-z0-9][a-z0-9-]{0,31}``, must not collide with the
        reserved chat builtins, and must be unique across other skills.
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

    def delete(self, skill_id: str) -> bool:
        with self._lock:
            data = self._read()
            new_data = [item for item in data if item.get("id") != skill_id]
            if len(new_data) == len(data):
                return False
            self._write(new_data)
            return True

    def _read(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            content = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(content, list):
            return [item for item in content if isinstance(item, dict)]
        return []

    def _write(self, data: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self._path)
