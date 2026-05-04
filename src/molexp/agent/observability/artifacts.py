"""Inline artifact normalization (spec §6.5).

Tools may return raw dictionaries shaped like
``{"kind": "plot"|"table"|"text", ...}``. This module turns them into
:class:`ArtifactRef` objects so the UI does not need to know which
tool authored them.
"""

from __future__ import annotations

from typing import Any

from molexp.agent.types import ArtifactRef


_ARTIFACT_KINDS = {"plot", "table", "text", "file"}


def normalize_artifact(payload: Any) -> ArtifactRef | None:
    """Return an :class:`ArtifactRef` if ``payload`` is artifact-shaped."""

    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in _ARTIFACT_KINDS:
        return None
    return ArtifactRef(
        kind=kind,
        title=payload.get("title", ""),
        payload={k: v for k, v in payload.items() if k not in {"kind", "title", "path"}},
        path=payload.get("path"),
    )


def normalize_artifacts(payload: Any) -> tuple[ArtifactRef, ...]:
    """Normalize a value or iterable of artifact-shaped payloads."""

    if isinstance(payload, list):
        out = [normalize_artifact(p) for p in payload]
        return tuple(p for p in out if p is not None)
    single = normalize_artifact(payload)
    return (single,) if single else ()
