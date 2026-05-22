"""Production wiring for the capability probe.

:func:`build_capability_probe` reads the molmcp MCP-server entry from the
workspace's :class:`~molexp.agent.mcp.store.McpStore` and constructs a
:class:`~molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
bound to it. PlanMode calls this lazily on first ``run()`` when no probe
was injected — so ``import molexp.agent`` stays free of ``pydantic_ai``
(this module lives behind the ``_pydanticai/`` firewall).

When no molmcp server is configured, the function returns ``None`` and
PlanMode falls back to the fail-closed
:class:`~molexp.agent.modes.plan.capability_probe_null.NullCapabilityProbe`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mollog import get_logger

from molexp.agent._pydanticai.capability_probe import (
    _DEFAULT_MAX_GROUNDING_ITERATIONS,
    PydanticAICapabilityProbe,
    PydanticAiModel,
)

__all__ = ["build_capability_probe"]

_LOG = get_logger(__name__)

_MOLMCP_SERVER_NAME = "molmcp"
"""The seeded MCP-server name the probe attaches to (see
:mod:`molexp.agent.mcp.defaults`)."""


def build_capability_probe(
    *,
    workspace: Path | None,
    model: object,
    max_grounding_iterations: int = _DEFAULT_MAX_GROUNDING_ITERATIONS,
) -> PydanticAICapabilityProbe | None:
    """Build the molmcp-backed probe, or ``None`` when molmcp is absent.

    Args:
        workspace: Workspace root used to resolve the MCP store; the
            user-scope store is still consulted when this is ``None``.
        model: pydantic-ai model the probe's agents run on — passed as
            ``object`` so callers outside ``_pydanticai/`` (PlanMode)
            need not import the pydantic-ai model alias.
        max_grounding_iterations: Re-draft budget forwarded to the probe
            (a need whose every ``api_ref`` failed verification is
            re-drafted, bounded by this). Defaults to the probe's own
            default.

    Returns:
        A :class:`PydanticAICapabilityProbe` bound to the molmcp server,
        or ``None`` when no molmcp ``stdio`` entry is configured.
    """
    try:
        from molexp.agent.mcp.store import McpStore

        store = McpStore(workspace if workspace is not None else Path())
        entries = store.list()
    except OSError as exc:  # pragma: no cover — read-only fs / schema drift
        _LOG.warning(f"[capability-probe] could not read MCP store: {exc!r}")
        return None

    for entry in entries:
        if (
            entry.name == _MOLMCP_SERVER_NAME
            and entry.transport == "stdio"
            and entry.valid
            and not entry.shadowed
            and entry.command
        ):
            _LOG.debug(f"[capability-probe] wiring molmcp probe via {entry.command!r}")
            return PydanticAICapabilityProbe(
                model=cast("PydanticAiModel", model),
                molmcp_command=entry.command,
                molmcp_args=tuple(entry.args),
                max_grounding_iterations=max_grounding_iterations,
            )
    _LOG.debug("[capability-probe] no molmcp stdio server configured")
    return None
