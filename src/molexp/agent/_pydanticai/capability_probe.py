"""``PydanticAICapabilityProbe`` — the production capability probe.

The molmcp-backed :class:`~molexp.agent.modes.plan.protocols.CapabilityProbe`
implementation PlanMode's ``ExploreCapabilities`` stage uses in
production. It wraps two ``pydantic_ai.Agent``\\ s behind the narrowed
``probe(*, intent) -> ProbeResult`` protocol:

* a no-tool *needs drafter* that maps a typed
  :class:`~molexp.agent.modes._planning.IntentSpec` into a set of
  :class:`~molexp.agent.modes.plan.capability_evidence.DraftedNeed`\\ s;
* an MCP-attached *evidence gatherer* — ``Agent(toolsets=[MCPServerStdio(...)])``
  — run once per drafted need (each call independently budgeted) to
  resolve it via the project's source-introspection tools; the per-need
  results fold into one
  :class:`~molexp.agent.modes.plan.capability_evidence.CapabilityEvidenceBatch`.

The two structured outputs are folded into a single :class:`ProbeResult`.

**Behavioural constraint** (agent-layer charter): no hand-rolled MCP
dispatch loop, no manual tool-call iteration. The only allowed shape is
``Agent(...)`` construction + ``await agent.run(...)`` + ``async with
agent`` for connection management — pydantic-ai drives the agent ↔ MCP
loop end-to-end (tool listing, call dispatch, retries, output parsing).

This module is a sanctioned ``import pydantic_ai`` site under
``agent/_pydanticai/`` (alongside ``router.py`` / ``mcp.py``); see
CLAUDE.md § Layer charters for the firewall rule.

**Gap note**: pydantic-ai does not cover *capability projection* (flat
evidence → typed ``CapabilityGraph``) or *on-disk plan persistence* —
those stay molexp-owned (``capability_projection.py`` / ``PlanFolder``).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import cast

from mollog import get_logger
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.usage import UsageLimits

from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    CapabilityEvidenceItem,
    DraftedNeed,
)
from molexp.agent.modes.plan.protocols import ProbeResult

__all__ = ["PydanticAICapabilityProbe"]

_LOG = get_logger(__name__)

# pydantic-ai's ``Agent(model=...)`` accepts any of these shapes.
type PydanticAiModel = models.Model | models.KnownModelName | str

_DEFAULT_DISCOVERY_RETRIES = 3
"""``output_retries`` cascade for the discovery agent — covers transient
MCP subprocess restarts, single-shot tool-call failures, and one
structured-output validation slip."""

_DEFAULT_PER_NEED_REQUEST_LIMIT = 30
"""``request_limit`` for resolving ONE drafted need. Discovery runs the
agent once per need (not once per batch): each need gets its own bounded
budget, so one over-eager need cannot starve the rest and a failure
degrades only that need. With the economical discovery prompt a need
resolves in ~15-25 tool-call rounds on a thorough model; 30 leaves
headroom to reach ``final_result`` while still terminating a runaway."""


@contextlib.contextmanager
def _silence_process_stdio() -> Iterator[None]:
    """Temporarily point stdout/stderr fds at ``os.devnull``.

    Some MCP servers print startup banners directly from the child
    process; redirecting Python's ``sys.stderr`` is not enough because
    the child inherits OS-level file descriptors.
    """
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


# ── Structured agent-output schemas ────────────────────────────────────────


class _DraftedNeedsReport(BaseModel):
    """Structured output of the no-tool needs-drafting agent.

    Attributes:
        needs: The capabilities the plan plausibly requires.
        discovery_required: ``False`` short-circuits the MCP evidence
            pass entirely (pure-stdlib plan).
    """

    model_config = ConfigDict(extra="forbid")

    needs: tuple[DraftedNeed, ...] = ()
    discovery_required: bool = True


# ── System prompts ─────────────────────────────────────────────────────────


_NEEDS_SYSTEM_PROMPT = (
    "You are a capability-needs drafter. Given a typed IntentSpec, "
    "identify the project capabilities the plan will need from the "
    "project's source code. For each, return a stable need_id, a short "
    "capability description, a one-sentence rationale, candidate "
    "api_refs (fully-qualified, best guess), the need_ids it depends_on, "
    "any interchangeable alternatives, and needs_user_confirmation when "
    "using it has a user-visible consequence. Set discovery_required="
    "True when at least one need plausibly maps to a project symbol; "
    "set it False only for pure-stdlib plans. Do not write code."
)

_DISCOVERY_SYSTEM_PROMPT = (
    "You are a capability-discovery agent. You are given exactly ONE "
    "DraftedNeed. Confirm the capability exists in the project source "
    "and emit CapabilityEvidenceItem(s) for it: module, symbol, kind, "
    "signature, a one-line doc summary, a confidence score, and usage "
    "notes. Key every item by the need's need_id.\n"
    "Work economically — resolve the need with the FEWEST tool calls. "
    "Start from the drafted api_refs; use search_source or list_symbols "
    "to locate the symbol, then get_signature and get_docstring to "
    "confirm it. Do NOT read whole modules with get_source unless a "
    "signature is genuinely ambiguous, and never browse unrelated "
    "modules. Three to six tool calls is typical — stop as soon as you "
    "can name a concrete module + symbol + signature. "
    "Put any api_ref you cannot confirm in missing_refs rather than "
    "fabricating evidence. Match MCP tool parameter names exactly as "
    "the server documents them."
)


# ── Agent builders ─────────────────────────────────────────────────────────


def _build_needs_agent(model: PydanticAiModel) -> Agent[None, _DraftedNeedsReport]:
    """Construct the structured (tool-less) needs-drafting agent."""
    agent = Agent(
        model=model,
        output_type=_DraftedNeedsReport,
        system_prompt=_NEEDS_SYSTEM_PROMPT,
    )
    return cast("Agent[None, _DraftedNeedsReport]", agent)


def _build_discovery_agent(
    model: PydanticAiModel,
    *,
    server: MCPToolset,
    retries: int,
) -> Agent[None, CapabilityEvidenceBatch]:
    """Construct the MCP-attached evidence-gathering agent."""
    agent = Agent(
        model=model,
        output_type=CapabilityEvidenceBatch,
        system_prompt=_DISCOVERY_SYSTEM_PROMPT,
        toolsets=[server],
        output_retries=retries,
    )
    return cast("Agent[None, CapabilityEvidenceBatch]", agent)


# ── Probe ──────────────────────────────────────────────────────────────────


class PydanticAICapabilityProbe:
    """Concrete :class:`CapabilityProbe` built on two ``pydantic_ai.Agent``\\ s.

    Construction is cheap — SDK ``Agent`` construction does no IO. The
    MCP subprocess is spawned lazily on the first :meth:`probe` call and
    torn down at :meth:`aclose`.

    Args:
        model: pydantic-ai model used by both agents (typically the
            HEAVY tier model).
        molmcp_command: Executable for the molmcp MCP server.
        molmcp_args: Optional CLI args for the MCP server.
        molmcp_env: Optional environment overlay for the MCP server.
        retries: ``output_retries`` for the discovery agent.
        request_limit: Max model requests for resolving ONE drafted
            need (one request per MCP tool-call round-trip). Discovery
            runs the agent once per need with this budget each.
    """

    def __init__(
        self,
        *,
        model: PydanticAiModel,
        molmcp_command: str,
        molmcp_args: tuple[str, ...] = (),
        molmcp_env: dict[str, str] | None = None,
        retries: int = _DEFAULT_DISCOVERY_RETRIES,
        request_limit: int = _DEFAULT_PER_NEED_REQUEST_LIMIT,
    ) -> None:
        self._model = model
        self._molmcp_command = molmcp_command
        self._molmcp_args = molmcp_args
        self._molmcp_env = molmcp_env
        self._retries = retries
        self._request_limit = request_limit
        self._needs_agent = _build_needs_agent(model)
        self._server = MCPToolset(
            StdioTransport(
                command=molmcp_command,
                args=list(molmcp_args),
                env=molmcp_env,
            )
        )
        self._discovery_agent = _build_discovery_agent(model, server=self._server, retries=retries)

    async def aclose(self) -> None:
        """Tear-down hook — idempotent. The MCP server is managed per-call."""
        return

    async def __aenter__(self) -> PydanticAICapabilityProbe:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ── Protocol method ──────────────────────────────────────────────────

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        """Discover the capabilities ``intent`` needs (narrowed protocol).

        Runs the no-tool needs drafter, then — when discovery is
        warranted — the MCP-attached evidence gatherer. A failed MCP
        pass degrades to an empty evidence batch (the drafted needs are
        still returned, so plan synthesis can proceed and the plan-graph
        preflight fails the unevidenced bindings closed).
        """
        report = await self._draft_needs(intent)
        if not report.discovery_required or not report.needs:
            return ProbeResult(drafted_needs=report.needs)
        evidence = await self._discover(report.needs)
        return ProbeResult(drafted_needs=report.needs, evidence=evidence)

    # ── Internal stages ──────────────────────────────────────────────────

    async def _draft_needs(self, intent: IntentSpec) -> _DraftedNeedsReport:
        """Run the no-tool needs-drafting agent against the typed intent."""
        prompt = f"IntentSpec:\n{intent.model_dump_json(indent=2)}"
        result = await self._needs_agent.run(prompt)
        report = result.output
        _LOG.debug(
            f"[capability-probe] draft_needs needs={len(report.needs)} "
            f"discovery_required={report.discovery_required}"
        )
        return report

    async def _discover(self, needs: tuple[DraftedNeed, ...]) -> CapabilityEvidenceBatch:
        """Resolve evidence for each drafted need independently.

        The discovery agent runs **once per need**, each call bounded by
        its own ``request_limit``: an over-eager need cannot starve the
        rest, and a per-need failure degrades only that need (its
        api_refs land in ``missing_refs``). A failure spinning up the
        shared MCP session degrades the whole batch to empty — the
        caller still gets the drafted needs, and the plan-graph
        preflight fails the unevidenced bindings closed.
        """
        items: list[CapabilityEvidenceItem] = []
        missing: list[str] = []
        try:
            with _silence_process_stdio():
                async with self._discovery_agent:
                    for need in needs:
                        batch = await self._discover_one(need)
                        items.extend(batch.items)
                        missing.extend(batch.missing_refs)
        except Exception as exc:
            # Failure at the MCP-subprocess boundary (transport / spawn).
            # PlanMode's contract is "probe never raises"; degrade to an
            # empty batch and let the plan-graph preflight fail closed.
            _LOG.warning(
                f"[capability-probe] discovery MCP session failed: "
                f"{type(exc).__name__}: {exc}; degrading to empty evidence batch"
            )
            return CapabilityEvidenceBatch()
        _LOG.debug(
            f"[capability-probe] discover evidence={len(items)} "
            f"missing={len(missing)} over {len(needs)} need(s)"
        )
        return CapabilityEvidenceBatch(
            items=tuple(items),
            missing_refs=tuple(dict.fromkeys(missing)),
        )

    async def _discover_one(self, need: DraftedNeed) -> CapabilityEvidenceBatch:
        """Resolve evidence for one drafted need within its own budget.

        Any pydantic-ai / MCP / usage-limit failure degrades to marking
        the need's drafted ``api_refs`` missing — the sibling needs keep
        whatever evidence they resolved.
        """
        prompt = "DraftedNeed:\n" + need.model_dump_json()
        try:
            result = await self._discovery_agent.run(
                prompt,
                usage_limits=UsageLimits(request_limit=self._request_limit),
            )
            return result.output
        except Exception as exc:
            _LOG.warning(
                f"[capability-probe] need {need.need_id!r} discovery failed: "
                f"{type(exc).__name__}: {exc}; marking its api_refs missing"
            )
            return CapabilityEvidenceBatch(missing_refs=need.api_refs)
