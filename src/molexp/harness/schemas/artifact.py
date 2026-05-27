"""``ArtifactRef`` + ``ArtifactKind`` — the basic state unit of a harness run.

Per ``.claude/notes/harness-goal.md`` §4.1: every harness-tracked product
(user plan, experiment report, workflow IR, bound workflow, execution
result, …) is referenced by an ``ArtifactRef`` carrying its kind, URI,
content hash, lineage, and creator metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ArtifactKind", "ArtifactRef"]


ArtifactKind = Literal[
    "user_plan",
    "experiment_report",
    "workflow_ir",
    "bound_workflow",
    "test_spec",
    "execution_plan",
    "execution_result",
    "test_result",
    "analysis_result",
    "final_report",
    "audit_report",
    "stdout",
    "stderr",
    "log",
    "input_file",
    "output_file",
    "plot",
    "dataset",
    "checkpoint",
    "validation_report",
]


class ArtifactRef(BaseModel):
    """Reference to one harness artifact.

    ``sha256`` is **bare hex** (no ``sha256:`` prefix). The companion
    :class:`molexp.harness.store.file_artifact_store.FileArtifactStore`
    strips the prefix returned by :func:`molexp.workspace.utils.compute_content_hash`
    before populating this field; the format stays compatible with the
    workspace-layer ``Asset.content_hash`` (which retains the prefix) so a
    later convergence Phase can unify the two.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: ArtifactKind
    uri: str
    sha256: str
    created_at: datetime
    created_by: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _sha256_must_be_bare_hex(cls, value: str) -> str:
        if ":" in value:
            raise ValueError(
                "ArtifactRef.sha256 must be bare hex; strip the 'sha256:' "
                "prefix returned by compute_content_hash before assigning."
            )
        if not value or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("ArtifactRef.sha256 must be a hex digest")
        return value
