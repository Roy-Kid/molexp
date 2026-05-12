"""``PydanticAICapabilityProbe`` — pydantic-ai-native capability discovery.

Wraps two ``pydantic_ai.Agent`` instances behind the
:class:`~molexp.agent.modes.plan.protocols.CapabilityProbe` Protocol:

* :func:`build_needs_agent` — structured agent (no tools) that maps
  the plan brief into a
  :class:`~molexp.agent.modes.plan.capability.CapabilityNeedReport`.
  Runs *before* the workflow IR is compiled so its only input is the
  plan brief's natural-language stages.
* :func:`build_discovery_agent` — structured agent attached to the
  project MCP server through ``toolsets=[MCPServerStdio(...)]``. The
  pydantic-ai SDK drives the agent ↔ MCP loop end-to-end (tool listing,
  call dispatch, retries, output parsing); the
  :class:`CapabilityEvidenceBatch` it returns is consumed verbatim by
  ``CompileWorkflowIR`` / ``CompileTaskIR`` to type the workflow IR.

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
from mcp import types as mcp_types
from mollog import get_logger
from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPServerStdio

from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeedReport,
)
from molexp.agent.modes.plan.context import PlanRepairContext
from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired
from molexp.agent.modes.plan.schemas import PlanBrief

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
    "You are a capability-needs drafter. Given the user goal and the "
    "current plan context, identify the project capabilities each step "
    "will need from the project's source code.\n\n"
    "For each required capability, return a short capability "
    "description, a one-sentence rationale, an expected_kind hint "
    "(class / callable / module / constant / protocol / namespace), "
    "and optional query_hints biasing the downstream MCP search. Use "
    "the natural-language stage label (or any other identifier present "
    "in the input) as the task_id field — at this point in the "
    "pipeline the IR's task_ids do not yet exist, so a stable stage "
    "label is the source of truth.\n\n"
    "Set discovery_required=True when at least one step plausibly "
    "needs a project symbol the code generator does not already know "
    "about. Set discovery_required=False only for pure-stdlib paths.\n\n"
    "Do not write code. Do not design TaskIO yet. Only describe what "
    "the project must supply for the plan to be implementable."
)


_DISCOVERY_SYSTEM_PROMPT = (
    "You are a capability-discovery agent. For every CapabilityNeed in "
    "the input report, use the project's MCP source-introspection "
    "tools to find modules, functions, classes, signatures, docstrings, "
    "and return shapes.\n\n"
    "Every user prompt includes an \"Available MCP tools\" block with "
    "the live tool list fetched from the MCP server at runtime — it "
    "documents every tool's name, parameters, and types. Always match "
    "parameter names exactly as listed there; never guess or substitute "
    "parameter names between tools.\n\n"
    "The input may also include discovery hints. A required hint means "
    "the user explicitly constrained the implementation; if you cannot "
    "produce evidence for it, record a MissingCapability rather than "
    "falling back silently. A preferred hint should be queried first, "
    "but fallback is allowed when you record why. A hint-strength row "
    "only biases query terms.\n\n"
    "Emit one CapabilityEvidence per resolved need. The evidence must "
    "be sufficient for downstream nodes to decide implementation "
    "strategy and Task input / output types — fill in namespace, "
    "package, module, symbol, kind, signature, usage_notes, "
    "doc_summary, and api_ref (where api_ref equals "
    "f'{module}.{symbol}'). Preserve relevant hints and tracked "
    "namespaces in the returned batch.\n\n"
    "Record any need you cannot satisfy as a MissingCapability with "
    "the appropriate mcp_no_match / mcp_low_confidence / mcp_timeout "
    "reason; do not fabricate evidence to fill the gap. Set "
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
    command: str = "",
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    server: MCPServerStdio | None = None,
    retries: int = _DEFAULT_DISCOVERY_RETRIES,
) -> tuple[Agent[None, CapabilityEvidenceBatch], MCPServerStdio]:
    """Construct the MCP-attached evidence-gathering agent.

    Args:
        model: pydantic-ai model id string or model object.
        command: Executable for the MCP server (the ``command`` field
            from the molmcp :class:`~molexp.agent.mcp.store.StdioSpec`).
            Ignored when ``server`` is provided.
        args: Optional CLI args appended to ``command``.
        env: Optional environment overlay.
        server: Pre-built ``MCPServerStdio``. When provided, ``command``
            / ``args`` / ``env`` are ignored.
        retries: ``output_retries=`` cascade applied to ``Agent`` so
            transient MCP failures or low-confidence outputs trigger a
            retry inside pydantic-ai. Defaults to 3.

    Returns:
        Tuple of ``(Agent[None, CapabilityEvidenceBatch], MCPServerStdio)``.
        The caller should keep the server reference for preflight
        ``list_tools()`` calls.
    """
    if server is None:
        server = MCPServerStdio(command=command, args=list(args), env=env)
    agent = Agent(
        model=model,
        output_type=CapabilityEvidenceBatch,
        system_prompt=_DISCOVERY_SYSTEM_PROMPT,
        toolsets=[server],
        retries=retries,
    )
    return cast("Agent[None, CapabilityEvidenceBatch]", agent), server


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
        self._discovery_agent, self._mcp_server = build_discovery_agent(
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
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport:
        """Run the needs-drafting agent against the plan brief.

        The user prompt is a YAML-rendered plan brief. pydantic-ai
        parses the LLM response into a :class:`CapabilityNeedReport`
        directly via ``output_type``.
        """
        prompt = _render_needs_prompt(plan_brief, repair_context=repair_context)
        _LOG.debug(f"[capability-probe] draft_needs prompt_chars={len(prompt)}")
        result = await self._needs_agent.run(prompt)
        report = result.output
        _LOG.debug(
            "[capability-probe] draft_needs done "
            f"discovery_required={report.discovery_required} needs={len(report.needs)}"
        )
        return report

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        """Run the MCP-attached evidence-gathering agent.

        Returns an empty + ``discovery_skipped=True`` batch when the
        upstream report did not request discovery — the discovery agent
        is never instantiated against the MCP server in that case
        (pure-stdlib short-circuit).

        For non-skipped runs, fetches the live MCP tool list via
        ``self._mcp_server.list_tools()`` so the prompt carries the
        exact parameter names and types the server exposes — no
        hardcoded signatures that drift.
        """
        if not report.discovery_required:
            _LOG.debug("[capability-probe] discover skipped — discovery_required=False")
            return CapabilityEvidenceBatch(
                evidence=(),
                missing=(),
                discovery_skipped=True,
            )

        try:
            tools = await self._mcp_server.list_tools()
            _LOG.debug(
                f"[capability-probe] discover preflight tools={len(tools)}"
            )
        except Exception:
            _LOG.warning(
                "[capability-probe] discover preflight failed — "
                "MCP server may not be running; proceeding without tool list",
                exc_info=True,
            )
            tools = []

        prompt = _render_discovery_prompt(
            report,
            tools=tools,
            repair_context=repair_context,
        )
        _LOG.debug(
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
        _LOG.debug(
            "[capability-probe] discover done "
            f"evidence={len(batch.evidence)} missing={len(batch.missing)}"
        )
        return batch


# ── Prompt rendering ──────────────────────────────────────────────────────


def _format_tool_preflight(tools: list[mcp_types.Tool]) -> str:
    """Render the live MCP tool list as a compact prompt block.

    Each tool gets one line: ``name(param: type, ...) — description``.
    The parameter list is derived from ``inputSchema.properties`` (when
    present) so the model sees the exact parameter names the server
    expects — no hardcoded signatures that drift when molmcp adds or
    renames tools.
    """
    lines = ["## Available MCP tools (live from server)\n"]
    for tool in tools:
        params = _format_tool_params(tool.inputSchema)
        desc = tool.description or ""
        # Keep description to first sentence for brevity.
        short_desc = desc.split(". ", 1)[0].strip()
        if short_desc and not short_desc.endswith("."):
            short_desc += "."
        lines.append(f"- `{tool.name}({params})` — {short_desc}")
    return "\n".join(lines)


def _format_tool_params(schema: dict) -> str:
    """Extract a compact ``name: type`` parameter list from a JSON Schema."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if not props:
        return ""
    parts: list[str] = []
    for name, prop in props.items():
        type_str = _json_type_name(prop)
        if name in required:
            parts.append(f"{name}: {type_str}")
        else:
            parts.append(f"{name}: {type_str} | None")
    return ", ".join(parts)


def _json_type_name(prop: dict) -> str:
    """Map a JSON Schema property to a short type name."""
    type_val = prop.get("type", "any")
    if isinstance(type_val, list):
        # e.g. ["string", "null"] → "str | None"
        names = [_json_scalar(t) for t in type_val if t != "null"]
        if "null" in type_val:
            return f"{' | '.join(names)} | None"
        return " | ".join(names)
    return _json_scalar(type_val)


def _json_scalar(type_name: str) -> str:
    """Shorten JSON Schema type names to familiar Python-like names."""
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }.get(type_name, type_name)


def _render_needs_prompt(
    plan_brief: PlanBrief,
    *,
    repair_context: PlanRepairContext | None = None,
) -> str:
    """Bundle the plan brief into a YAML prompt for the needs agent.

    Discovery runs before IR compilation so the plan brief is the only
    upstream artefact at this point in the pipeline; the needs agent
    drafts one :class:`CapabilityNeed` per stage that plausibly requires
    project code. pydantic-ai feeds the YAML string to the model verbatim.
    """
    payload = plan_brief.model_dump(mode="json")
    block = repair_context.prompt_block(node_id="DraftCapabilityNeeds") if repair_context else ""
    if block:
        payload["repair_context"] = block
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
    )


def _render_discovery_prompt(
    report: CapabilityNeedReport,
    *,
    tools: list[mcp_types.Tool],
    repair_context: PlanRepairContext | None = None,
) -> str:
    """Render the report as a JSON document for the discovery agent.

    Prepends a live ``Available MCP tools`` block built from *tools*
    (fetched via ``MCPServerStdio.list_tools()`` at runtime) so the
    model always sees the exact parameter names and types the server
    exposes.

    JSON (rather than YAML) keeps the per-need fields tightly packed so
    the model sees one need per document line — easier to plan
    individual tool calls against.
    """
    tool_block = _format_tool_preflight(tools)
    payload = report.model_dump(mode="json")
    block = repair_context.prompt_block(node_id="DiscoverCapabilities") if repair_context else ""
    if block:
        payload["repair_context"] = block
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"{tool_block}\n\n{body}"
