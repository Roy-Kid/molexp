"""``PydanticAICapabilityProbe`` — the production capability probe.

The molmcp-backed :class:`~molexp.agent.modes.plan.protocols.CapabilityProbe`
implementation PlanMode's ``ExploreCapabilities`` stage uses in
production. It runs a **draft → ground** pipeline behind the narrowed
``probe(*, intent) -> ProbeResult`` protocol:

* a no-tool *needs drafter* maps a typed
  :class:`~molexp.agent.modes._planning.IntentSpec` into a set of
  :class:`~molexp.agent.modes.plan.capability_evidence.DraftedNeed`\\ s;
* a *grounding* stage verifies every drafted ``api_ref`` against the real
  source through an MCP-attached agent and folds the per-ref verdicts
  into a
  :class:`~molexp.agent.modes.plan.capability_evidence.CapabilityEvidenceBatch`;
* a need whose every ``api_ref`` failed verification is re-drafted with
  the rejection fed back to the no-tool drafter, bounded by
  ``max_grounding_iterations``.

Two-tier verify
===============

molmcp is an *index*: broad and fast, but lossy — a re-exported symbol
(``molpy.Atomistic``, re-exported from ``molpy.core.atomistic``) surfaces
as ``module`` / ``example`` hits, not a clean ``class`` hit. So the
grounding agent verifies each ref in two tiers:

* **Tier 1 — index query** (``search_source`` / ``list_symbols``): a
  clean ``class`` / ``function`` / ``callable`` hit resolves the ref.
* **Tier 2 — follow references**: when Tier 1 is inconclusive (only
  ``module`` / ``example`` hits, or nothing), the agent escalates —
  ``get_source`` the candidate module and the package ``__init__.py``,
  trace re-exports and the import chain to the real definition. A ref is
  ``missing`` only when *both* tiers miss.

**Behavioural constraint** (agent-layer charter): no hand-rolled MCP
dispatch loop, no manual tool-call iteration. The only allowed shape is
``Agent(...)`` construction + ``await agent.run(...)`` + ``async with
agent`` for connection management — pydantic-ai drives the agent ↔ MCP
loop end-to-end.

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
from collections.abc import Awaitable, Callable, Iterator
from typing import cast

from mollog import get_logger
from pydantic import BaseModel, ConfigDict, Field
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

_DEFAULT_GROUNDING_RETRIES = 3
"""``output_retries`` for the grounding agent — covers transient MCP
subprocess restarts and one structured-output validation slip."""

_DEFAULT_PER_NEED_REQUEST_LIMIT = 30
"""``request_limit`` for grounding ONE drafted need. The grounding agent
runs once per need; each need gets its own bounded budget so one
over-eager need cannot starve the rest, and a per-need failure degrades
only that need."""

_DEFAULT_MAX_GROUNDING_ITERATIONS = 2
"""Re-draft budget: a need whose every ``api_ref`` failed verification is
re-drafted with the rejection fed back, up to this many extra rounds.
``0`` disables re-draft — one verification pass and done."""


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
        discovery_required: ``False`` short-circuits the grounding pass
            entirely (pure-stdlib plan).
    """

    model_config = ConfigDict(extra="forbid")

    needs: tuple[DraftedNeed, ...] = ()
    discovery_required: bool = True


class _RefVerdict(BaseModel):
    """One drafted ``api_ref``'s grounding verdict.

    The grounding agent emits one of these per drafted ref. ``resolved``
    records whether the ref maps to a real symbol after the two-tier
    verify; the canonical ``module`` / ``symbol`` / ``kind`` /
    ``signature`` are filled in when ``resolved`` is true.

    Attributes:
        need_id: The :class:`DraftedNeed` this ref belongs to.
        api_ref: The drafted reference that was checked.
        resolved: Whether both verify tiers located a real symbol.
        module: Canonical module of the resolved symbol.
        symbol: Canonical symbol name.
        kind: ``class`` / ``function`` / ``callable`` / ``module`` / ….
        signature: The symbol's signature, if applicable.
        doc_summary: A one-line docstring summary.
        confidence: Confidence in the match, in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    need_id: str
    api_ref: str
    resolved: bool
    module: str = ""
    symbol: str = ""
    kind: str = ""
    signature: str = ""
    doc_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _GroundingReport(BaseModel):
    """The grounding agent's structured output for one drafted need.

    Carries one :class:`_RefVerdict` per ``api_ref`` the agent verified.
    """

    model_config = ConfigDict(extra="forbid")

    verdicts: tuple[_RefVerdict, ...] = ()


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

_GROUNDING_SYSTEM_PROMPT = (
    "You are a capability-grounding agent. You are given exactly ONE "
    "DraftedNeed. For EVERY api_ref it lists, decide whether the ref "
    "names a symbol that really exists in the project source, and emit "
    "exactly one verdict per api_ref.\n"
    "Verify each ref in TWO tiers:\n"
    "  Tier 1 — index query: use search_source / list_symbols to look "
    "the ref up. A clean class / function / callable hit resolves it — "
    "record resolved=true with the canonical module, symbol, kind and "
    "signature (use get_signature / get_docstring to confirm).\n"
    "  Tier 2 — follow references: if Tier 1 returns only module or "
    "example hits, or nothing, do NOT conclude the ref is missing. "
    "Escalate — get_source the candidate module AND the package "
    "__init__.py, follow re-exports ('from .x import Y') and the import "
    "chain to the symbol's real definition. Many real symbols are "
    "re-exported at a package's top level: a class queried by its bare "
    "name surfaces as a module hit, and get_source on the __init__.py "
    "confirms it.\n"
    "Record resolved=false for a ref ONLY when both tiers fail to locate "
    "a real symbol. Work economically — a handful of tool calls per ref. "
    "Match MCP tool parameter names exactly as the server documents "
    "them. Do not write code."
)

_REDRAFT_SYSTEM_PROMPT = (
    "You are a capability-needs re-drafter. You are given DraftedNeeds "
    "whose candidate api_refs were checked against the project source "
    "and found NOT to exist. For each, draft corrected api_refs that "
    "name real project symbols — keep the same need_id, capability, "
    "rationale, depends_on, alternatives and needs_user_confirmation; "
    "replace only the api_refs, and never reuse a rejected ref. If you "
    "cannot name a plausible real symbol, return the need with empty "
    "api_refs rather than inventing one. Do not write code."
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


def _build_redraft_agent(model: PydanticAiModel) -> Agent[None, _DraftedNeedsReport]:
    """Construct the tool-less needs re-drafting agent."""
    agent = Agent(
        model=model,
        output_type=_DraftedNeedsReport,
        system_prompt=_REDRAFT_SYSTEM_PROMPT,
    )
    return cast("Agent[None, _DraftedNeedsReport]", agent)


def _build_grounding_agent(
    model: PydanticAiModel,
    *,
    toolsets: tuple[MCPToolset, ...] = (),
    tools: tuple[Callable[..., object], ...] = (),
    retries: int = _DEFAULT_GROUNDING_RETRIES,
) -> Agent[None, _GroundingReport]:
    """Construct the two-tier source-introspection grounding agent.

    ``toolsets`` carries the molmcp ``MCPToolset`` in production; ``tools``
    lets tests inject plain fake source-introspection callables instead
    of spawning a real MCP subprocess.
    """
    agent = Agent(
        model=model,
        output_type=_GroundingReport,
        system_prompt=_GROUNDING_SYSTEM_PROMPT,
        toolsets=list(toolsets),
        tools=list(tools),
        output_retries=retries,
    )
    return cast("Agent[None, _GroundingReport]", agent)


# ── Pure grounding logic ───────────────────────────────────────────────────


def _join_ref(module: str, symbol: str) -> str:
    """Join ``module`` + ``symbol`` into a dotted ref, tolerating blanks."""
    return ".".join(part for part in (module, symbol) if part)


def _evidence_item(verdict: _RefVerdict) -> CapabilityEvidenceItem:
    """Build a :class:`CapabilityEvidenceItem` from a resolved verdict."""
    return CapabilityEvidenceItem(
        need_id=verdict.need_id,
        api_ref=verdict.api_ref or _join_ref(verdict.module, verdict.symbol),
        module=verdict.module,
        symbol=verdict.symbol,
        kind=verdict.kind,
        signature=verdict.signature,
        doc_summary=verdict.doc_summary,
        confidence=verdict.confidence,
    )


def _fold_grounding(
    needs: tuple[DraftedNeed, ...],
    verdicts: tuple[_RefVerdict, ...],
) -> tuple[tuple[DraftedNeed, ...], CapabilityEvidenceBatch]:
    """Fold per-ref grounding verdicts into grounded needs + an evidence batch.

    Pure — no LLM, no I/O. A ``resolved`` verdict becomes a
    :class:`CapabilityEvidenceItem`; an unresolved one's ``api_ref`` lands
    in ``missing_refs``. A need with at least one resolved ref has its
    ``api_refs`` narrowed to the resolved subset; a need with none keeps
    its original drafted refs so the projection classifies it ``missing``
    — fail-closed.

    Args:
        needs: The drafted (or re-drafted) needs.
        verdicts: One or more :class:`_RefVerdict`\\ s per need's refs.

    Returns:
        ``(grounded_needs, evidence_batch)``.
    """
    by_need: dict[str, list[_RefVerdict]] = {}
    for verdict in verdicts:
        by_need.setdefault(verdict.need_id, []).append(verdict)

    items: list[CapabilityEvidenceItem] = []
    missing: list[str] = []
    grounded: list[DraftedNeed] = []
    for need in needs:
        resolved_refs: list[str] = []
        for verdict in by_need.get(need.need_id, ()):
            if verdict.resolved:
                items.append(_evidence_item(verdict))
                resolved_refs.append(verdict.api_ref)
            else:
                missing.append(verdict.api_ref)
        if resolved_refs:
            grounded.append(need.model_copy(update={"api_refs": tuple(resolved_refs)}))
        else:
            grounded.append(need)
    return tuple(grounded), CapabilityEvidenceBatch(
        items=tuple(items),
        missing_refs=tuple(dict.fromkeys(missing)),
    )


def _needs_to_redraft(
    needs: tuple[DraftedNeed, ...],
    verdicts: tuple[_RefVerdict, ...],
) -> tuple[DraftedNeed, ...]:
    """Return needs that have ``api_refs`` but zero resolved verdicts — pure.

    A need with no ``api_refs`` is left alone: re-drafting cannot help a
    need that names no candidate symbol.
    """
    resolved_need_ids = {v.need_id for v in verdicts if v.resolved}
    return tuple(need for need in needs if need.api_refs and need.need_id not in resolved_need_ids)


def _replace_needs(
    needs: tuple[DraftedNeed, ...],
    replacements: tuple[DraftedNeed, ...],
) -> tuple[DraftedNeed, ...]:
    """Return ``needs`` with any same-``need_id`` entry swapped for its
    replacement — pure, order-preserving."""
    by_id = {need.need_id: need for need in replacements}
    return tuple(by_id.get(need.need_id, need) for need in needs)


def _merge_verdicts(
    old: tuple[_RefVerdict, ...],
    refreshed_needs: tuple[DraftedNeed, ...],
    fresh: tuple[_RefVerdict, ...],
) -> tuple[_RefVerdict, ...]:
    """Replace ``old`` verdicts for re-verified needs with ``fresh`` ones — pure."""
    refreshed_ids = {need.need_id for need in refreshed_needs}
    kept = tuple(v for v in old if v.need_id not in refreshed_ids)
    return kept + fresh


def _all_unresolved(needs: tuple[DraftedNeed, ...]) -> tuple[_RefVerdict, ...]:
    """Build all-unresolved verdicts for every ``api_ref`` of ``needs``."""
    return tuple(
        _RefVerdict(need_id=need.need_id, api_ref=ref, resolved=False)
        for need in needs
        for ref in need.api_refs
    )


def _coerce_verdicts(need: DraftedNeed, report: _GroundingReport) -> tuple[_RefVerdict, ...]:
    """Normalize the agent's report to exactly one verdict per ``api_ref``.

    The agent's verdicts are matched to the need's drafted refs; a ref
    the agent did not rule on is treated as unresolved. Every verdict's
    ``need_id`` and ``api_ref`` are forced to the drafted values.
    """
    by_ref = {verdict.api_ref: verdict for verdict in report.verdicts}
    out: list[_RefVerdict] = []
    for ref in need.api_refs:
        verdict = by_ref.get(ref)
        if verdict is None:
            out.append(_RefVerdict(need_id=need.need_id, api_ref=ref, resolved=False))
        else:
            out.append(verdict.model_copy(update={"need_id": need.need_id, "api_ref": ref}))
    return tuple(out)


def _render_redraft_prompt(
    failed_needs: tuple[DraftedNeed, ...],
    verdicts: tuple[_RefVerdict, ...],
) -> str:
    """Render the re-draft user prompt — failed needs + their rejected refs."""
    rejected: dict[str, list[str]] = {}
    for verdict in verdicts:
        if not verdict.resolved:
            rejected.setdefault(verdict.need_id, []).append(verdict.api_ref)
    blocks: list[str] = []
    for need in failed_needs:
        refs = ", ".join(rejected.get(need.need_id, list(need.api_refs))) or "(none)"
        blocks.append(
            f"need_id={need.need_id}\n"
            f"  capability: {need.capability}\n"
            f"  rejected api_refs (do NOT reuse): {refs}\n"
            f"  full need: {need.model_dump_json()}"
        )
    return "Re-draft these needs with corrected api_refs:\n\n" + "\n\n".join(blocks)


# ── Grounding loop ─────────────────────────────────────────────────────────


_VerifyFn = Callable[[tuple[DraftedNeed, ...]], Awaitable[tuple[_RefVerdict, ...]]]
_RedraftFn = Callable[
    [tuple[DraftedNeed, ...], tuple[_RefVerdict, ...]],
    Awaitable[tuple[DraftedNeed, ...]],
]


async def _grounding_loop(
    needs: tuple[DraftedNeed, ...],
    *,
    verify: _VerifyFn,
    redraft: _RedraftFn,
    max_iterations: int,
) -> tuple[tuple[DraftedNeed, ...], CapabilityEvidenceBatch]:
    """Verify → (bounded) re-draft → re-verify, then fold.

    ``verify`` grounds a set of needs' refs; ``redraft`` re-drafts the
    needs that came back fully unresolved. Bounded by ``max_iterations``
    extra rounds; on budget exhaustion the still-unresolved needs keep
    their drafted refs and :func:`_fold_grounding` classifies them
    ``missing`` (fail-closed). Parameterized over ``verify`` / ``redraft``
    so the loop is unit-tested without a live agent or MCP server.
    """
    working = needs
    verdicts: tuple[_RefVerdict, ...] = await verify(working)
    for _ in range(max(max_iterations, 0)):
        failed = _needs_to_redraft(working, verdicts)
        if not failed:
            break
        redrafted = await redraft(failed, verdicts)
        working = _replace_needs(working, redrafted)
        fresh = await verify(redrafted)
        verdicts = _merge_verdicts(verdicts, redrafted, fresh)
    return _fold_grounding(working, verdicts)


# ── Probe ──────────────────────────────────────────────────────────────────


class PydanticAICapabilityProbe:
    """Concrete :class:`CapabilityProbe` — draft → ground over molmcp.

    Construction is cheap — SDK ``Agent`` construction does no IO. The
    MCP subprocess is spawned lazily on the first :meth:`probe` call and
    torn down per verification pass.

    Args:
        model: pydantic-ai model the agents run on (typically the HEAVY
            tier model).
        molmcp_command: Executable for the molmcp MCP server.
        molmcp_args: Optional CLI args for the MCP server.
        molmcp_env: Optional environment overlay for the MCP server.
        retries: ``output_retries`` for the grounding agent.
        request_limit: Max model requests for grounding ONE drafted need.
        max_grounding_iterations: Re-draft budget — a need whose every
            ``api_ref`` failed verification is re-drafted with the
            rejection fed back, up to this many extra rounds. ``0``
            disables re-draft.
    """

    def __init__(
        self,
        *,
        model: PydanticAiModel,
        molmcp_command: str,
        molmcp_args: tuple[str, ...] = (),
        molmcp_env: dict[str, str] | None = None,
        retries: int = _DEFAULT_GROUNDING_RETRIES,
        request_limit: int = _DEFAULT_PER_NEED_REQUEST_LIMIT,
        max_grounding_iterations: int = _DEFAULT_MAX_GROUNDING_ITERATIONS,
    ) -> None:
        self._model = model
        self._molmcp_command = molmcp_command
        self._molmcp_args = molmcp_args
        self._molmcp_env = molmcp_env
        self._request_limit = request_limit
        self._max_grounding_iterations = max_grounding_iterations
        self._needs_agent = _build_needs_agent(model)
        self._redraft_agent = _build_redraft_agent(model)
        self._server = MCPToolset(
            StdioTransport(
                command=molmcp_command,
                args=list(molmcp_args),
                env=molmcp_env,
            )
        )
        self._grounding_agent = _build_grounding_agent(
            model, toolsets=(self._server,), retries=retries
        )

    @property
    def max_grounding_iterations(self) -> int:
        """The bounded re-draft budget configured for this probe."""
        return self._max_grounding_iterations

    async def aclose(self) -> None:
        """Tear-down hook — idempotent. The MCP server is managed per-call."""
        return

    async def __aenter__(self) -> PydanticAICapabilityProbe:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ── Protocol method ──────────────────────────────────────────────────

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        """Discover + ground the capabilities ``intent`` needs.

        Runs the no-tool needs drafter, then — when discovery is
        warranted — the two-tier grounding loop. A failed MCP pass
        degrades to all-unresolved (the drafted needs are still returned,
        so plan synthesis can proceed and the plan-graph preflight fails
        the unevidenced bindings closed).
        """
        report = await self._draft_needs(intent)
        if not report.discovery_required or not report.needs:
            return ProbeResult(drafted_needs=report.needs)
        grounded_needs, evidence = await _grounding_loop(
            report.needs,
            verify=self._verify_pass,
            redraft=self._redraft_pass,
            max_iterations=self._max_grounding_iterations,
        )
        return ProbeResult(drafted_needs=grounded_needs, evidence=evidence)

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

    async def _verify_pass(self, needs: tuple[DraftedNeed, ...]) -> tuple[_RefVerdict, ...]:
        """Two-tier verify every need's refs through the grounding agent.

        Runs the MCP-attached grounding agent once per need (each call
        independently budgeted). A failure at the MCP-subprocess boundary
        degrades the whole pass to all-unresolved verdicts — the probe's
        contract is "never raise"; the plan-graph preflight then fails
        the unevidenced bindings closed.
        """
        verdicts: list[_RefVerdict] = []
        try:
            with _silence_process_stdio():
                async with self._grounding_agent:
                    for need in needs:
                        if not need.api_refs:
                            continue
                        verdicts.extend(await self._verify_one(need))
        except Exception as exc:
            _LOG.warning(
                f"[capability-probe] grounding MCP session failed: "
                f"{type(exc).__name__}: {exc}; degrading to all-unresolved"
            )
            return _all_unresolved(needs)
        _LOG.debug(
            f"[capability-probe] verify_pass needs={len(needs)} "
            f"verdicts={len(verdicts)} resolved={sum(v.resolved for v in verdicts)}"
        )
        return tuple(verdicts)

    async def _verify_one(self, need: DraftedNeed) -> tuple[_RefVerdict, ...]:
        """Ground one need's refs within its own request budget.

        Any pydantic-ai / MCP / usage-limit failure degrades to marking
        every ref of this need unresolved — sibling needs keep theirs.
        """
        prompt = "DraftedNeed:\n" + need.model_dump_json()
        try:
            result = await self._grounding_agent.run(
                prompt,
                usage_limits=UsageLimits(request_limit=self._request_limit),
            )
            return _coerce_verdicts(need, result.output)
        except Exception as exc:
            _LOG.warning(
                f"[capability-probe] need {need.need_id!r} grounding failed: "
                f"{type(exc).__name__}: {exc}; marking its api_refs unresolved"
            )
            return _all_unresolved((need,))

    async def _redraft_pass(
        self,
        failed_needs: tuple[DraftedNeed, ...],
        verdicts: tuple[_RefVerdict, ...],
    ) -> tuple[DraftedNeed, ...]:
        """Re-draft the fully-unresolved needs, feeding back the rejected refs.

        A failure degrades to returning the needs unchanged — they stay
        unresolved and :func:`_fold_grounding` classifies them ``missing``.
        """
        prompt = _render_redraft_prompt(failed_needs, verdicts)
        try:
            result = await self._redraft_agent.run(prompt)
        except Exception as exc:
            _LOG.warning(
                f"[capability-probe] re-draft failed: "
                f"{type(exc).__name__}: {exc}; keeping needs as drafted"
            )
            return failed_needs
        return result.output.needs
