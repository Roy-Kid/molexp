"""``ArtifactStore`` Protocol.

The structural contract every artifact backend implements: harness layers
program against this Protocol, never against a concrete class, so the
backend (filesystem, blob store, object store) can be swapped without
touching call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from molexp.harness.schemas import ArtifactKind, ArtifactRef

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
    ) -> ArtifactRef: ...

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef: ...

    def put_file(
        self,
        kind: ArtifactKind,
        path: Path,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef: ...

    def get(self, artifact_id: str) -> bytes: ...

    def get_ref(self, artifact_id: str) -> ArtifactRef: ...

    def list_by_kind(self, kind: ArtifactKind) -> list[ArtifactRef]: ...

    def latest_by_kind(self, kind: ArtifactKind) -> ArtifactRef | None: ...
