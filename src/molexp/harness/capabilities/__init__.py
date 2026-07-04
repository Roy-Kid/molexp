"""Built-in ``ToolCapability`` catalogs shipped directly by the harness.

Unlike the molmcp-discovered science toolchain, these capabilities are static
built-ins registered directly onto the registry (link 05). Currently the
workspace-curation catalog (:func:`curation_capabilities`, exposing
``molexp.workspace.curation.*``) and the run-lifecycle catalog
(:func:`lifecycle_capabilities`, the five gated run verbs) — every entry gated
automatically by its declared ``side_effects``.
"""

from __future__ import annotations

from molexp.harness.capabilities.curation import (
    CURATION_CAPABILITIES,
    curation_capabilities,
)
from molexp.harness.capabilities.lifecycle import (
    LIFECYCLE_CAPABILITIES,
    lifecycle_capabilities,
)

__all__ = [
    "CURATION_CAPABILITIES",
    "LIFECYCLE_CAPABILITIES",
    "curation_capabilities",
    "lifecycle_capabilities",
]
