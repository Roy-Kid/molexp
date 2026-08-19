"""Agent operations surface — protocols + **builtin** tools.

Hard invariant (**auto-discovery law**):

* Upstream packages (molpy, molpack, molplot, …) and molmcp catalogs must
  **not** be mirrored as hand-maintained tool lists in molexp.
* Session-open / turn-open **enumeration** (MCP tool catalog, discovery
  index) is the only source of third-party capability names.
* **Builtin** tools: Chat surface (inspect + scratch code + discover) by
  default; full surface adds ``workspace_ensure`` / ``run_land`` for archive.
* **Surface keys** (:mod:`molexp.agent.ops.surface`) are the declared
  coeffect spec; MCP mutators are classified, not deny-listed.
* Science and plotting always go through :class:`CodeEnv` (write + run
  Python that imports upstream packages).

Public surface::

    from molexp.agent.ops import (
        AgentSessionContext,
        build_session_context,
        build_ops_tools,
        BUILTIN_TOOL_NAMES,
        CHAT_TOOL_NAMES,
        DEFAULT_OPS_PREAMBLE,
    )
"""

from __future__ import annotations

from molexp.agent.ops.builtins import (
    ARCHIVE_TOOL_NAMES,
    BUILTIN_SOURCE,
    BUILTIN_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    FULL_TOOL_NAMES,
    builtin_tool_specs,
)
from molexp.agent.ops.context import AgentSessionContext, build_session_context
from molexp.agent.ops.lifecycle import LIFECYCLE_TOOL_NAMES, lifecycle_tools
from molexp.agent.ops.preamble import CHAT_OPS_PREAMBLE, DEFAULT_OPS_PREAMBLE, FULL_OPS_PREAMBLE
from molexp.agent.ops.protocols import (
    BehaviorPolicy,
    CodeEnv,
    Discovery,
    StructureOps,
)
from molexp.agent.ops.surface import (
    CHAT_SURFACE,
    FULL_SURFACE,
    LIFECYCLE_SURFACE,
    SurfaceKey,
    ToolSurface,
    surface_for_mode,
)
from molexp.agent.ops.tools import build_ops_tools, render_discovery_catalog

__all__ = [
    "ARCHIVE_TOOL_NAMES",
    "BUILTIN_SOURCE",
    "BUILTIN_TOOL_NAMES",
    "CHAT_OPS_PREAMBLE",
    "CHAT_SURFACE",
    "CHAT_TOOL_NAMES",
    "DEFAULT_OPS_PREAMBLE",
    "FULL_OPS_PREAMBLE",
    "FULL_SURFACE",
    "FULL_TOOL_NAMES",
    "LIFECYCLE_SURFACE",
    "LIFECYCLE_TOOL_NAMES",
    "AgentSessionContext",
    "BehaviorPolicy",
    "CodeEnv",
    "Discovery",
    "StructureOps",
    "SurfaceKey",
    "ToolSurface",
    "build_ops_tools",
    "build_session_context",
    "builtin_tool_specs",
    "lifecycle_tools",
    "render_discovery_catalog",
    "surface_for_mode",
]
