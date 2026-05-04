"""Legacy plugin shell — kept for the MCP submodules during Phase 4 cutover.

Per spec §9 the agent runtime, types, tools, tool registry, and the
``_pydantic_ai/`` private package are removed in Phase 3. The
remaining MCP-related modules (``mcp_oauth``, ``mcp_probe``,
``mcp_store``) and a small handful of helpers (``commands``,
``policy``) stay until Phase 4 promotes them into
``molexp.plugins.tool_mcp``. Skills, native tools, the model client,
and the provider config now live in :mod:`molexp.agent` and
:mod:`molexp.plugins.model_pydanticai`.
"""

from __future__ import annotations
