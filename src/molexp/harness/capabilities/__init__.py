"""Built-in ``ToolCapability`` catalogs shipped directly by the harness.

Unlike the molmcp-discovered science toolchain, these capabilities are static
built-ins registered directly onto the registry (link 05). Currently the
workspace-curation catalog: :func:`curation_capabilities` returns the
``ToolCapability`` entries exposing ``molexp.workspace.curation.*`` as
harness-invokable tools, gated automatically by their declared ``side_effects``.
"""

from __future__ import annotations

from molexp.harness.capabilities.curation import (
    CURATION_CAPABILITIES,
    curation_capabilities,
)

__all__ = ["CURATION_CAPABILITIES", "curation_capabilities"]
