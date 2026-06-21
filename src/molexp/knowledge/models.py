"""``meta.yaml`` model for the OKF ``molexp.knowledge`` layer.

In OKF each Concept is a directory split into two physical files: the
structured ``meta.yaml`` (this model) and the narrative ``index.md``
(plain markdown, owned by ``Folder`` in okf-01-03). They never mix —
there is no YAML frontmatter inside ``index.md``.

:class:`ConceptMeta` is the base model for ``meta.yaml``. Per OKF the only
required field is ``type``; everything else is optional. Concept subtypes
(Run, Experiment, …) add their own structured fields (``config_hash``,
settled ``status``, ``params``, ``parameter_space``, children pointers);
because this base model is used to read any subtype's ``meta.yaml``, it
preserves unknown keys losslessly on round-trip (``extra="allow"``) so a
base-model read never truncates a subtype's fields.

High-frequency machine state (heartbeats, owner PIDs, resume seeds, cache)
never goes in ``meta.yaml`` — it lives in the per-Concept ``_ops/``
operational sidecar (okf-01-03). ``meta.yaml`` carries only settled,
human-meaningful structured knowledge; no narrative blob fields.
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
    (``extra="allow"``); the model is frozen (immutable after
    construction), matching the repo's pure-data-type convention.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    id: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None

    @classmethod
    def from_yaml(cls, text: str) -> ConceptMeta:
        """Parse a ``meta.yaml`` string into a :class:`ConceptMeta`.

        Uses ``yaml.safe_load``; an empty document yields a validation
        error (``type`` is required).
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
