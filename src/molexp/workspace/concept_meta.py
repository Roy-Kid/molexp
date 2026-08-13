"""``ConceptMeta`` — the structured ``meta.json`` payload of an OKF Concept.

Workspace-local OKF base for typed concept heads (e.g.
:class:`molexp.workspace.reference_meta.ReferenceMeta`). The agent layer keeps
its own ``AgentMeta`` / ``AgentSessionMeta`` shapes; this base serves the
workspace-owned Concept types. Kept in the workspace layer so nothing here
depends on ``molexp.knowledge`` (which is now the concept-type registry only).

Serialization is **JSON only** (same format family as entity ``*.json``).
"""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConceptMeta(BaseModel):
    """Structured ``meta.json`` payload of one OKF Concept.

    Attributes:
        type: Concept subtype discriminator — the one required OKF field.
        id: Optional stable identifier (path is the canonical identity).
        tags: Optional categorical labels.
        timestamp: Optional last-update timestamp.

    Subtype-specific keys are accepted and preserved verbatim
    (``extra="allow"``); the model is frozen (immutable after construction),
    matching the repo's pure-data-type convention.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    id: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None

    @classmethod
    def from_json(cls, text: str) -> ConceptMeta:
        """Parse a ``meta.json`` string into a :class:`ConceptMeta`."""
        data = json.loads(text) if text.strip() else {}
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a ``meta.json`` string (pretty, stable key order)."""
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"

    # ── Back-compat aliases (YAML-era names) ──────────────────────────────

    @classmethod
    def from_yaml(cls, text: str) -> ConceptMeta:
        """Deprecated alias for :meth:`from_json` (also accepts legacy YAML text)."""
        stripped = text.lstrip()
        if stripped.startswith(("{", "[")):
            return cls.from_json(text)
        import yaml

        data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Deprecated alias for :meth:`to_json`."""
        return self.to_json()


__all__ = ["ConceptMeta"]
