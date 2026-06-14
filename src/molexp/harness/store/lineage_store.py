"""``ArtifactLineageStore`` Protocol — pipeline-artifact lineage, nothing more.

Scope (deliberately narrow): this store records **which pipeline stage
produced which artifact, derived from which prior artifact** — the
``derived_from`` edge graph between harness :class:`ArtifactRef` ids, each
edge optionally stamped with the producing ``stage`` name and the pipeline
``run_id`` so a lineage chain is traversable end-to-end.

It is NOT a general provenance system. Run-level provenance — parameters,
merged config + ``config_hash``, profile, workflow identity, execution
history, environment — is owned by :mod:`molexp.workspace`
(``RunMetadata`` / ``AssetCatalog`` / ``Asset.producer``). Code-version and
environment capture belong there, never here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from molexp.harness.schemas import ArtifactRef

__all__ = ["ArtifactLineageStore"]


@runtime_checkable
class ArtifactLineageStore(Protocol):
    """Structural type for any artifact-lineage-edge backend."""

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
        *,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> None: ...

    def trace_backward(self, artifact_id: str) -> list[ArtifactRef]: ...

    def trace_forward(self, artifact_id: str) -> list[ArtifactRef]: ...

    def lineage_graph(self, artifact_id: str) -> dict[str, Any]: ...
