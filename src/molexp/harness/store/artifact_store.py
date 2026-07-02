"""``ArtifactStore`` Protocol.

The structural contract every artifact backend implements: harness layers
program against this Protocol, never against a concrete class, so the
backend (filesystem, blob store, object store) can be swapped without
touching call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from molexp.harness.schemas import ArtifactKind, PlanArtifactRef

__all__ = ["ArtifactStore"]


@runtime_checkable
class ArtifactStore(Protocol):
    """Structural type for any artifact-content store."""

    def put_json(
        self,
        kind: ArtifactKind,
        obj: object,
        created_by: str,
        parent_ids: list[str],
    ) -> PlanArtifactRef: ...

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        created_by: str,
        parent_ids: list[str],
    ) -> PlanArtifactRef: ...

    def put_file(
        self,
        kind: ArtifactKind,
        path: Path,
        created_by: str,
        parent_ids: list[str],
    ) -> PlanArtifactRef: ...

    def get(self, artifact_id: str) -> bytes: ...

    def get_ref(self, artifact_id: str) -> PlanArtifactRef: ...

    def list_by_kind(self, kind: ArtifactKind) -> list[PlanArtifactRef]: ...

    def latest_by_kind(self, kind: ArtifactKind) -> PlanArtifactRef | None: ...
