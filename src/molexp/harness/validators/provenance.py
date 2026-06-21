"""Structural validator for artifact provenance lineage (Phase 5).

Given an ``artifact_id`` and the run's :class:`ArtifactStore` +
:class:`ArtifactLineageStore`, walks ``trace_backward`` and verifies the
artifact can be traced to an ancestor of the declared ``root_kind``
(default ``"user_plan"``).

Pure function — no I/O beyond the stores it receives; never raises.
``ArtifactNotFoundError`` from a missing ``get_ref`` lookup is caught and
surfaced as a ``ValidationViolation`` rather than propagated.

Three codes:

- ``artifact_not_found`` (error): the artifact itself doesn't exist in
  the store.
- ``unreachable_root`` (error): ``trace_backward`` returned ancestors,
  but none of them has ``kind == root_kind``.
- ``orphan_artifact`` (warning): ``trace_backward`` returned an empty
  list AND the artifact's own kind ≠ ``root_kind``. May be a partial run
  or an artifact written outside the harness pipeline.

Clean cases:

- The artifact's own kind IS the root_kind (empty trace_backward — it
  IS the root).
- ``trace_backward`` includes at least one ancestor with the root_kind.
"""

from __future__ import annotations

from molexp.harness.errors import ArtifactNotFoundError
from molexp.harness.schemas.artifact import ArtifactKind
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.lineage_store import ArtifactLineageStore

__all__ = ["ProvenanceValidator"]


class ProvenanceValidator:
    @staticmethod
    def validate(
        artifact_id: str,
        *,
        artifact_store: ArtifactStore,
        lineage_store: ArtifactLineageStore,
        root_kind: ArtifactKind = "user_plan",
    ) -> ValidationReport:
        violations: list[ValidationViolation] = []

        # 1. artifact_not_found — catch the typed error rather than bubbling it.
        try:
            ref = artifact_store.get_ref(artifact_id)
        except ArtifactNotFoundError as exc:
            violations.append(
                ValidationViolation(
                    code="artifact_not_found",
                    message=f"artifact id {artifact_id!r} not found in artifact_store ({exc})",
                    path="artifact_id",
                )
            )
            return ValidationReport.from_violations(
                target_kind="provenance",
                target_id=artifact_id,
                violations=violations,
            )

        # If the artifact itself IS the root_kind, no ancestors required.
        if ref.kind == root_kind:
            return ValidationReport.from_violations(
                target_kind="provenance",
                target_id=artifact_id,
                violations=violations,
            )

        ancestors = lineage_store.trace_backward(artifact_id)

        # 3. orphan_artifact (warning) — no ancestors AND not the root itself.
        if not ancestors:
            violations.append(
                ValidationViolation(
                    code="orphan_artifact",
                    message=(
                        f"artifact {artifact_id!r} (kind={ref.kind!r}) has no ancestors "
                        f"and is not of root kind {root_kind!r}"
                    ),
                    path="artifact_id",
                    severity="warning",
                )
            )
            return ValidationReport.from_violations(
                target_kind="provenance",
                target_id=artifact_id,
                violations=violations,
            )

        # 2. unreachable_root — has ancestors but none matches root_kind.
        if not any(ancestor.kind == root_kind for ancestor in ancestors):
            violations.append(
                ValidationViolation(
                    code="unreachable_root",
                    message=(
                        f"artifact {artifact_id!r} (kind={ref.kind!r}) has "
                        f"{len(ancestors)} ancestor(s) but none of kind {root_kind!r}"
                    ),
                    path="provenance.lineage",
                )
            )

        return ValidationReport.from_violations(
            target_kind="provenance",
            target_id=artifact_id,
            violations=violations,
        )
