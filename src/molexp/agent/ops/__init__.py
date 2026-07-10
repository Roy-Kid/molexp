"""Agent operations surface — protocols, not scattered tools.

Hard invariant (**auto-discovery law**):

* Upstream packages (molpy, molpack, molplot, …) and molmcp catalogs must
  **not** be mirrored as hand-maintained tool lists in molexp.
* Session-open / turn-open **enumeration** (MCP tool catalog, discovery
  index) is the only source of third-party capability names.
* Structure verbs (ensure project/experiment) are **this package's**
  workspace API — stable, few, idempotent — not science mirrors.
* Science and plotting always go through :class:`CodeEnv` (write + run
  Python that imports upstream packages).

Public surface::

    from molexp.agent.ops import (
        AgentSessionContext,
        build_session_context,
        build_ops_tools,
        DEFAULT_OPS_PREAMBLE,
    )
"""

from __future__ import annotations

from molexp.agent.ops.context import AgentSessionContext, build_session_context
from molexp.agent.ops.preamble import DEFAULT_OPS_PREAMBLE
from molexp.agent.ops.protocols import (
    BehaviorPolicy,
    CodeEnv,
    Discovery,
    StructureOps,
)
from molexp.agent.ops.tools import OPS_TOOL_NAMES, build_ops_tools

__all__ = [
    "DEFAULT_OPS_PREAMBLE",
    "OPS_TOOL_NAMES",
    "AgentSessionContext",
    "BehaviorPolicy",
    "CodeEnv",
    "Discovery",
    "StructureOps",
    "build_ops_tools",
    "build_session_context",
]
