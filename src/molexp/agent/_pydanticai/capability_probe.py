"""``PydanticAICapabilityProbe`` — the production capability probe.

The molmcp-backed :class:`~molexp.agent.modes.plan.protocols.CapabilityProbe`
implementation PlanMode's ``ExploreCapabilities`` stage uses in
production. It wraps two ``pydantic_ai.Agent``\\ s behind the narrowed
``probe(*, intent) -> ProbeResult`` protocol:

* a no-tool *needs drafter* that maps a typed
  :class:`~molexp.agent.modes._planning.IntentSpec` into a set of
  :class:`~molexp.agent.modes.plan.capability_evidence.DraftedNeed`\\ s;
* an MCP-attached *evidence gatherer* — ``Agent(toolsets=[MCPServerStdio(...)])``
  — that resolves those needs into a
  :class:`~molexp.agent.modes.plan.capability_evidence.CapabilityEvidenceBatch`
  via the project's source-introspection tools.

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

from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
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
    "You are a capability-discovery agent. For every DraftedNeed in the "
    "input, use the project's MCP source-introspection tools to resolve "
    "concrete API evidence: the module, symbol, kind, signature, a "
    "one-line doc summary, a confidence score, and usage notes / "
    "limits. Emit one CapabilityEvidenceItem per resolved api_ref, "
    "keyed by the originating need_id. Record any api_ref you cannot "
    "resolve in missing_refs rather than fabricating evidence. Match "
    "MCP tool parameter names exactly as the server documents them."
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
    """

    def __init__(
        self,
        *,
        model: PydanticAiModel,
        molmcp_command: str,
        molmcp_args: tuple[str, ...] = (),
        molmcp_env: dict[str, str] | None = None,
        retries: int = _DEFAULT_DISCOVERY_RETRIES,
    ) -> None:
        self._model = model
        self._molmcp_command = molmcp_command
        self._molmcp_args = molmcp_args
        self._molmcp_env = molmcp_env
        self._retries = retries
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
        """Run the MCP-attached evidence gatherer against the drafted needs.

        Any MCP / pydantic-ai failure degrades to an empty evidence
        batch — the caller still gets the drafted needs, and the
        downstream plan-graph preflight fails the unevidenced capability
        bindings closed.
        """
        prompt = "DraftedNeeds:\n" + "\n".join(need.model_dump_json() for need in needs)
        try:
            with _silence_process_stdio():
                async with self._discovery_agent:
                    result = await self._discovery_agent.run(prompt)
            _LOG.debug(
                f"[capability-probe] discover evidence={len(result.output.items)} "
                f"missing={len(result.output.missing_refs)}"
            )
            return result.output
        except Exception as exc:
            # The discovery agent sits at the MCP-subprocess boundary,
            # where transport / tool-dispatch / output-validation errors
            # of many shapes surface. PlanMode's contract is "probe
            # never raises"; degrade to an empty batch and let the
            # plan-graph preflight fail the bindings closed.
            _LOG.warning(
                f"[capability-probe] discover failed: {type(exc).__name__}: {exc}; "
                "degrading to empty evidence batch"
            )
            return CapabilityEvidenceBatch()
