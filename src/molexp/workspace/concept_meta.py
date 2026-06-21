"""``ConceptMeta`` — the structured ``meta.yaml`` payload of an OKF Concept.

Workspace-local OKF base for typed ``meta.yaml`` bodies (e.g.
:class:`molexp.workspace.reference_meta.ReferenceMeta`). The agent layer keeps
its own ``AgentMeta`` / ``AgentSessionMeta`` shapes; this base serves the
workspace-owned Concept types. Kept in the workspace layer so nothing here
depends on ``molexp.knowledge`` (which is now the concept-type registry only).
"""

from __future__ import annotations

from datetime import datetime

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConceptMeta(BaseModel):
    """Structured ``meta.yaml`` payload of one OKF Concept.

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
    def from_yaml(cls, text: str) -> ConceptMeta:
        """Parse a ``meta.yaml`` string into a :class:`ConceptMeta`.

        Uses ``yaml.safe_load``; an empty document yields a validation error
        (``type`` is required).
        """
        data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize to a ``meta.yaml`` string via ``yaml.safe_dump``.

        Includes any subtype ``extra`` keys; key order is preserved
        (``sort_keys=False``). Datetimes are rendered through
        ``model_dump(mode="json")`` (ISO strings).
        """
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


__all__ = ["ConceptMeta"]
