"""``ProvenanceStore`` Protocol — artifact lineage contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from molexp.harness.schemas import ArtifactRef

__all__ = ["ProvenanceStore"]


@runtime_checkable
class ProvenanceStore(Protocol):
    """Structural type for any provenance-edge backend."""

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
    ) -> None: ...

    def trace_backward(self, artifact_id: str) -> list[ArtifactRef]: ...

    def trace_forward(self, artifact_id: str) -> list[ArtifactRef]: ...

    def lineage_graph(self, artifact_id: str) -> dict[str, Any]: ...
