"""``PydanticAICapabilityProbe`` — pydantic-ai-native capability discovery (Phase 4).

Wraps two ``pydantic_ai.Agent`` instances behind the
:class:`~molexp.agent.modes.plan.protocols.CapabilityProbe` Protocol:

* :func:`build_needs_agent` — structured agent (no tools) that maps
  the plan brief / contract / per-task briefs into a
  :class:`~molexp.agent.modes.plan.capability.CapabilityNeedReport`.
* :func:`build_discovery_agent` — structured agent attached to the
  molmcp MCP server through ``toolsets=[MCPServerStdio(...)]``. The
  pydantic-ai SDK drives the agent ↔ MCP loop end-to-end (tool listing,
  call dispatch, retries, output parsing); the
  :class:`CapabilityEvidenceBatch` it returns is consumed verbatim.

**Behavioural constraint** (rectification spec, Phase 4): no
hand-rolled MCP dispatch loops, no manual tool-call iteration, no
``async for chunk in agent.stream(...)`` plumbing. The only allowed
shape is ``Agent(...)`` construction + ``await agent.run(...)`` +
``async with agent`` for connection management. If pydantic-ai
introduces a richer streaming API for MCP, swap to it inside this
file — never re-implement the loop in molexp.

This module is the second permitted ``import pydantic_ai`` site under
``agent/_pydanticai/`` (alongside ``router.py``); see CLAUDE.md
§ Layer charters for the firewall rule.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import cast

import yaml
from mollog import get_logger
from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPServerStdio

from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeedReport,
)
from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired
from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    TaskIRBrief,
    WorkflowContract,
)

# pydantic-ai's ``Agent(model=...)`` accepts any of these shapes.
# Mirror the alias router.py uses so the capability_probe surface matches.
type PydanticAiModel = models.Model | models.KnownModelName | str

__all__ = [
    "PydanticAICapabilityProbe",
    "build_discovery_agent",
    "build_needs_agent",
]

_LOG = get_logger(__name__)


_DEFAULT_DISCOVERY_RETRIES = 3
"""Default ``output_retries`` cascade for the discovery agent.

Three rounds covers transient MCP subprocess restarts, single-shot
tool-call failures, and one structured-output validation slip without
making the cap visibly long when every retry actually fails. Tunable
per call via :func:`build_discovery_agent`'s ``retries`` kwarg or
:class:`PydanticAICapabilityProbe`'s constructor."""


# ── System prompts ────────────────────────────────────────────────────────


_NEEDS_SYSTEM_PROMPT = (
    "You are a capability-needs drafter for a molecular-simulation "
    "experiment-planning system. Given a natural-language implementation "
    "plan, the typed workflow contract, and the per-task briefs, decide "
    "whether the experiment will need any Molcrafts (molpy / molexp / "
    "molvis / molpack / molnex / molq / mollog / molcfg) APIs that the "
    "code generator does not already know about.\n\n"
    "Set discovery_required=True only when at least one task plausibly "
    "needs a Molcrafts symbol (a class, callable, constant). Set "
    "discovery_required=False for pure-stdlib workflows. For each "
    "drafted need, return a short capability description, a one-sentence "
    "rationale, an expected_kind hint (class / callable / module / "
    "constant), and optional query_hints biasing the discovery search."
)


_DISCOVERY_SYSTEM_PROMPT = (
    "You are a capability-discovery agent. For every CapabilityNeed in "
    "the input report, use the molmcp tools (list_modules, list_symbols, "
    "search_source, get_signature, get_docstring) to find the Molcrafts "
    "symbol that satisfies the need. Emit one CapabilityEvidence per "
    "need (api_ref MUST equal f'{module}.{symbol}'). Record any need "
    "you cannot satisfy as a MissingCapability with the appropriate "
    "mcp_no_match / mcp_low_confidence / mcp_timeout reason. Set "
    "discovery_skipped=False on every batch you produce — the caller "
    "sets True only when discovery is short-circuited upstream."
)


# ── Agent builders ────────────────────────────────────────────────────────


def build_needs_agent(model: PydanticAiModel) -> Agent[None, CapabilityNeedReport]:
    """Construct the structured (tool-less) needs-drafting agent.

    Args:
        model: pydantic-ai model id string or model object (e.g.
            ``"deepseek:deepseek-v4-flash"``). Typically the
            :class:`~molexp.agent.router.ModelTier.HEAVY` tier model
            picked by :class:`AgentRunner`.

    Returns:
        ``Agent[None, CapabilityNeedReport]`` with the needs-drafter
        system prompt baked in. The agent has no tools — its only job
        is to translate a plain-text request into the structured
        report.
    """
    agent = Agent(
        model=model,
        output_type=CapabilityNeedReport,
        system_prompt=_NEEDS_SYSTEM_PROMPT,
    )
    return cast("Agent[None, CapabilityNeedReport]", agent)


def build_discovery_agent(
    model: PydanticAiModel,
    *,
    command: str,
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    retries: int = _DEFAULT_DISCOVERY_RETRIES,
) -> Agent[None, CapabilityEvidenceBatch]:
    """Construct the MCP-attached evidence-gathering agent.

    Args:
        model: pydantic-ai model id string or model object.
        command: Executable for the MCP server (the ``command`` field
            from the molmcp :class:`~molexp.agent.mcp.store.StdioSpec`).
        args: Optional CLI args appended to ``command``.
        env: Optional environment overlay.
        retries: ``output_retries=`` cascade applied to ``Agent`` so
            transient MCP failures or low-confidence outputs trigger a
            retry inside pydantic-ai. Defaults to 3.

    Returns:
        ``Agent[None, CapabilityEvidenceBatch]`` with the molmcp MCP
        server attached as a toolset. The agent loop, tool listing,
        per-tool dispatch, and structured-output parsing are all driven
        by pydantic-ai natively.
    """
    server = MCPServerStdio(command=command, args=list(args), env=env)
    agent = Agent(
        model=model,
        output_type=CapabilityEvidenceBatch,
        system_prompt=_DISCOVERY_SYSTEM_PROMPT,
        toolsets=[server],
        retries=retries,
    )
    return cast("Agent[None, CapabilityEvidenceBatch]", agent)


# ── Probe ────────────────────────────────────────────────────────────────


class PydanticAICapabilityProbe:
    """Concrete :class:`CapabilityProbe` impl built on two ``pydantic_ai.Agent``\\ s.

    Construction is cheap — both agents are built eagerly because the
    SDK construction itself does no IO. The MCP subprocess is spawned
    lazily on the first :meth:`discover` call and torn down at probe
    close (:meth:`aclose`).

    Args:
        model: pydantic-ai model used by both agents (typically the
            HEAVY tier model).
        molmcp_command: Executable for the molmcp MCP server.
        molmcp_args: Optional CLI args.
        molmcp_env: Optional environment overlay.
        retries: ``output_retries`` for the discovery agent. Defaults
            to 3.
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
        self._needs_agent = build_needs_agent(model)
        self._discovery_agent = build_discovery_agent(
            model,
            command=molmcp_command,
            args=molmcp_args,
            env=molmcp_env,
            retries=retries,
        )
        self._stack: AsyncExitStack | None = None

    async def aclose(self) -> None:
        """Tear down the MCP subprocess if it was started.

        Idempotent — safe to call when no session was ever opened.
        """
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    async def __aenter__(self) -> PydanticAICapabilityProbe:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ── Protocol methods ──────────────────────────────────────────────────

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
        contract: WorkflowContract,
        briefs: tuple[TaskIRBrief, ...],
    ) -> CapabilityNeedReport:
        """Run the needs-drafting agent.

        The user prompt is a YAML-rendered bundle of the plan brief +
        contract + briefs. pydantic-ai parses the LLM response into a
        :class:`CapabilityNeedReport` directly via ``output_type``.
        """
        prompt = _render_needs_prompt(plan_brief, contract, briefs)
        _LOG.info(f"[capability-probe] draft_needs prompt_chars={len(prompt)}")
        result = await self._needs_agent.run(prompt)
        report = result.output
        _LOG.info(
            "[capability-probe] draft_needs done "
            f"discovery_required={report.discovery_required} needs={len(report.needs)}"
        )
        return report

    async def discover(self, report: CapabilityNeedReport) -> CapabilityEvidenceBatch:
        """Run the MCP-attached evidence-gathering agent.

        Returns an empty + ``discovery_skipped=True`` batch when the
        upstream report did not request discovery — the discovery agent
        is never instantiated against the MCP server in that case
        (pure-stdlib short-circuit).

        For non-skipped runs, opens an :class:`AsyncExitStack` over
        ``async with self._discovery_agent`` so the MCP subprocess
        survives a single :meth:`discover` call. The pydantic-ai SDK
        drives the entire tool-call loop; molexp does not iterate over
        needs or invoke MCP tools by hand.
        """
        if not report.discovery_required:
            _LOG.info("[capability-probe] discover skipped — discovery_required=False")
            return CapabilityEvidenceBatch(
                evidence=(),
                missing=(),
                discovery_skipped=True,
            )
        prompt = _render_discovery_prompt(report)
        _LOG.info(
            f"[capability-probe] discover start needs={len(report.needs)} "
            f"prompt_chars={len(prompt)}"
        )
        try:
            async with self._discovery_agent:
                result = await self._discovery_agent.run(prompt)
        except Exception as exc:
            # Catching `Exception` here is intentional: this is the
            # pydantic-ai / MCP SDK boundary, where any number of
            # transport, tool-dispatch, output-validation, or OAuth
            # errors can surface. The repair loop's only signal is
            # ``CapabilityDiscoveryRequired``; collapse all of them
            # into that shape (``__cause__`` preserves the original
            # for debug).
            _LOG.warning(
                f"[capability-probe] discover failed: {type(exc).__name__}: {exc}",
                exc_info=True,
            )
            raise CapabilityDiscoveryRequired(
                "Discovery agent failed; pipeline cannot continue.",
                reason="mcp_error",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        batch = result.output
        _LOG.info(
            "[capability-probe] discover done "
            f"evidence={len(batch.evidence)} missing={len(batch.missing)}"
        )
        return batch


# ── Prompt rendering ──────────────────────────────────────────────────────


def _render_needs_prompt(
    plan_brief: PlanBrief,
    contract: WorkflowContract,
    briefs: tuple[TaskIRBrief, ...],
) -> str:
    """Bundle the plan brief, contract, and per-task briefs into one prompt.

    YAML keeps the structure readable and matches how the rest of the
    pipeline persists artefacts; pydantic-ai feeds the string to the
    model verbatim.
    """
    payload = {
        "plan_brief": plan_brief.model_dump(mode="json"),
        "workflow_contract": contract.model_dump(mode="json"),
        "task_briefs": [b.model_dump(mode="json") for b in briefs],
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _render_discovery_prompt(report: CapabilityNeedReport) -> str:
    """Render the report as a JSON document for the discovery agent.

    JSON (rather than YAML) keeps the per-need fields tightly packed so
    the model sees one need per document line — easier to plan
    individual tool calls against.
    """
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
