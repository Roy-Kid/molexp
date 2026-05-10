"""Capability discovery schemas + codegen-evidence validator (Phase 3).

Owns the five frozen pydantic models the PlanMode capability-discovery
pipeline uses plus a pure-AST validator that codegen nodes invoke
after writing each generated module to disk.

pydantic-ai does **not** provide a capability-discovery layer:
pydantic-ai's remit is model-side execution (tool dispatch / MCP /
retries / message history / structured output). The PlanMode workflow
adds an LLM-driven *need-drafting* step followed by an
*evidence-gathering* step driven through an MCP server (molmcp), and
the resulting evidence has to round-trip through codegen so the
validator can refuse unevidenced API usage. That orchestration is
molexp's own; the discovery agent itself uses pydantic-ai natively
(see ``_pydanticai/capability_probe.py`` in Phase 4).
"""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "MOLCRAFTS_NAMESPACES",
    "CapabilityEvidence",
    "CapabilityEvidenceBatch",
    "CapabilityNeed",
    "CapabilityNeedReport",
    "MissReason",
    "MissingCapability",
    "validate_codegen_evidence",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")

MOLCRAFTS_NAMESPACES: tuple[str, ...] = (
    "molpy",
    "molexp",
    "molvis",
    "molpack",
    "molnex",
    "molq",
    "mollog",
    "molcfg",
)
"""Top-level package prefixes the capability validator scans for.

Used as the namespace filter inside :func:`validate_codegen_evidence`
and the future ``ValidateWorkspace.capability_evidence_check`` (Phase
6). Adding a namespace here only expands validation scope; it does
**not** switch discovery on or off — that decision is owned by the
LLM in ``DraftCapabilityNeeds``."""


MissReason = Literal[
    "mcp_no_match",
    "mcp_low_confidence",
    "mcp_timeout",
    "unevidenced_in_code",
    "undeclared_in_code",
    "declared_but_unused",
]
"""Closed enumeration of the six valid :attr:`MissingCapability.reason`
values. The first three are emitted by the discovery agent
(``DiscoverCapabilities``); the last three by
:func:`validate_codegen_evidence` after codegen."""


# ── Frozen pydantic models ─────────────────────────────────────────────────


class CapabilityNeed(BaseModel):
    """One unit of capability needed by an experiment task.

    Drafted by ``DraftCapabilityNeeds`` from the per-task IR briefs and
    persisted under ``capability/needs.yaml``.

    Attributes:
        task_id: Experiment-task id this need belongs to (matches the
            corresponding ``TaskIRBrief.task_id``).
        capability: Natural-language description (one phrase) of what
            the task needs — e.g. ``"construct a peptide from amino-acid codes"``.
        rationale: One sentence saying *why* the capability is needed
            for the task to succeed.
        expected_kind: Hint about what shape the symbol should take.
            Canonical strings: ``"class"`` / ``"callable"`` / ``"module"`` /
            ``"constant"`` / ``"protocol"`` / ``"namespace"``. Unknown
            values are not enforced at the type level (different MCP
            servers may surface other kinds).
        query_hints: Optional query keywords passed to the discovery
            agent's MCP search call to bias the result.
    """

    model_config = _FROZEN

    task_id: str
    capability: str
    rationale: str = ""
    expected_kind: str = ""
    query_hints: tuple[str, ...] = ()


class CapabilityNeedReport(BaseModel):
    """``DraftCapabilityNeeds`` output.

    Attributes:
        discovery_required: ``False`` lets the rest of the pipeline
            short-circuit discovery entirely (e.g. when the LLM is sure
            the task is pure-stdlib). ``True`` forces ``DiscoverCapabilities``
            to actually run the MCP-attached agent.
        needs: Drafted needs; may be empty when ``discovery_required`` is
            ``False``.
        rationale_summary: One paragraph explaining the discovery
            decision; persisted verbatim into ``capability/needs.yaml``
            so a human reviewer can audit the gate.
    """

    model_config = _FROZEN

    discovery_required: bool
    needs: tuple[CapabilityNeed, ...] = ()
    rationale_summary: str = ""


class CapabilityEvidence(BaseModel):
    """Resolved evidence for one need.

    Attributes:
        need_fingerprint: Stable identifier of the originating
            :class:`CapabilityNeed` — typically ``f"{task_id}:{capability}"``.
            Lets discovery results match back to their need without
            embedding the full need object.
        source: Discovery channel that produced this evidence (e.g.
            ``"molmcp"``). Future probes may surface ``"rag"`` /
            ``"docs"`` / etc.
        package: Top-level Molcrafts package (must be one of
            :data:`MOLCRAFTS_NAMESPACES`).
        module: Fully-qualified module path (``"molpy.builders.peptide"``).
        symbol: Symbol name within the module (``"PeptideBuilder"``).
        kind: Symbol kind (``"class"`` / ``"callable"`` / etc.).
        signature: Canonical Python signature line.
        doc_summary: First-paragraph summary of the symbol's docstring.
        api_ref: Canonical Python dotted-path identifier, equivalent to
            ``f"{module}.{symbol}"``. **Primary key for the
            codegen-evidence diff** in :func:`validate_codegen_evidence`.
        confidence: ``0.0..1.0`` confidence score returned by the
            discovery agent.
    """

    model_config = _FROZEN

    need_fingerprint: str
    source: str
    package: str
    module: str
    symbol: str
    kind: str
    signature: str
    doc_summary: str = ""
    api_ref: str
    confidence: float = 1.0


class MissingCapability(BaseModel):
    """One row in the missing-capability ledger.

    Produced by both the discovery agent (for ``mcp_*`` reasons) and by
    :func:`validate_codegen_evidence` (for ``unevidenced_in_code`` /
    ``undeclared_in_code`` / ``declared_but_unused``).

    Attributes:
        need: Originating :class:`CapabilityNeed`. ``None`` for
            validator-emitted misses where no upstream need maps to the
            offending ref (the diff is keyed on ``api_ref`` only).
        reason: One of the six values defined by :data:`MissReason`;
            constrained at the type level.
        detail: Human-readable description; for validator-emitted misses
            includes the offending ``api_ref`` so the repair loop can
            re-target discovery.
        repairable: Whether the repair loop should retry. ``False``
            signals a permanent miss (no MCP server can resolve this).
    """

    model_config = _FROZEN

    need: CapabilityNeed | None = None
    reason: MissReason
    detail: str = ""
    repairable: bool = True


class CapabilityEvidenceBatch(BaseModel):
    """``DiscoverCapabilities`` output.

    Attributes:
        evidence: All resolved evidence rows.
        missing: Needs that could not be evidenced — usually from the
            ``mcp_*`` family (no match / low confidence / timeout).
        discovery_skipped: Set when ``discovery_required`` was ``False``
            upstream; tells codegen + validator to relax the
            ``__capability_evidence__`` block requirement entirely.
    """

    model_config = _FROZEN

    evidence: tuple[CapabilityEvidence, ...] = ()
    missing: tuple[MissingCapability, ...] = ()
    discovery_skipped: bool = False


# ── AST validator ──────────────────────────────────────────────────────────


_DECLARED_BLOCK_NAME = "__capability_evidence__"


def validate_codegen_evidence(
    source: str,
    batch: CapabilityEvidenceBatch,
) -> tuple[MissingCapability, ...]:
    """Diff a generated module's evidence claims against the discovery batch.

    The validator walks ``source`` with :mod:`ast` and returns one
    :class:`MissingCapability` per (ref, reason) the diff turns up.
    ``batch.discovery_skipped`` short-circuits the entire routine
    (returns ``()``) — pure-stdlib codegen paths are exempt from the
    gate.

    The four diff branches (verbatim from the spec):

    a. ``ast_refs - evidence_refs``        → ``unevidenced_in_code``
    b. ``declared_refs - evidence_refs``   → ``unevidenced_in_code``
    c. ``ast_refs - declared_refs``        → ``undeclared_in_code``
    d. ``declared_refs - ast_refs``        → ``declared_but_unused``

    where:

    - ``ast_refs`` is the set of dotted paths reconstructed from
      :class:`ast.ImportFrom` modules + maximal :class:`ast.Attribute`
      chains whose leftmost :class:`ast.Name` matches a
      :data:`MOLCRAFTS_NAMESPACES` prefix.
    - ``declared_refs`` is the set of strings inside the module-level
      ``__capability_evidence__: tuple[str, ...] = (...)`` literal,
      extracted via :func:`ast.literal_eval`.
    - ``evidence_refs`` is ``{e.api_ref for e in batch.evidence}``.

    Args:
        source: Full module source text (post-codegen, post-compile).
        batch: Discovery output the generated module is supposed to
            match.

    Returns:
        Tuple of :class:`MissingCapability` rows; empty if the diff is
        clean or ``batch.discovery_skipped`` is True.
    """
    if batch.discovery_skipped:
        return ()

    tree = ast.parse(source)
    declared_refs = _extract_declared_refs(tree)
    ast_refs = _extract_ast_refs(tree)
    evidence_refs = {e.api_ref for e in batch.evidence}

    misses: list[MissingCapability] = []

    # Spec branches (a) and (b) merge: every ref absent from evidence
    # is reported once whether it came from AST or the declared block.
    unevidenced = (ast_refs | declared_refs) - evidence_refs
    for ref in sorted(unevidenced):
        misses.append(
            MissingCapability(
                need=None,
                reason="unevidenced_in_code",
                detail=f"{ref} not in evidence batch",
                repairable=True,
            )
        )

    undeclared = ast_refs - declared_refs
    for ref in sorted(undeclared):
        misses.append(
            MissingCapability(
                need=None,
                reason="undeclared_in_code",
                detail=f"{ref} used in AST but missing from __capability_evidence__",
                repairable=True,
            )
        )

    declared_unused = declared_refs - ast_refs
    for ref in sorted(declared_unused):
        misses.append(
            MissingCapability(
                need=None,
                reason="declared_but_unused",
                detail=f"{ref} declared in __capability_evidence__ but not used",
                repairable=True,
            )
        )

    return tuple(misses)


def _extract_declared_refs(tree: ast.Module) -> set[str]:
    """Pull ``__capability_evidence__: tuple[str, ...] = (...)`` literals.

    Handles both the annotated (:class:`ast.AnnAssign`) and plain
    (:class:`ast.Assign`) forms; only module-level assignments count.
    Returns the empty set when no such assignment exists or the value
    is not a string-tuple literal.
    """
    refs: set[str] = set()
    for node in tree.body:
        value = _declared_block_value(node)
        if value is not None:
            refs.update(_literal_string_tuple(value))
    return refs


def _declared_block_value(node: ast.stmt) -> ast.expr | None:
    """Return the RHS expression if ``node`` is a module-level
    ``__capability_evidence__`` assignment (annotated or plain), else None."""
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == _DECLARED_BLOCK_NAME
    ):
        return node.value
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == _DECLARED_BLOCK_NAME for t in node.targets
    ):
        return node.value
    return None


def _literal_string_tuple(node: ast.expr) -> set[str]:
    """Evaluate ``node`` as ``tuple[str, ...]`` via :func:`ast.literal_eval`.

    Returns the empty set if the literal is not a tuple of strings or
    if :func:`ast.literal_eval` raises.
    """
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(value, tuple):
        return set()
    return {item for item in value if isinstance(item, str)}


def _extract_ast_refs(tree: ast.Module) -> set[str]:
    """Reconstruct Molcrafts-prefixed dotted paths from imports + attribute chains.

    Walks every :class:`ast.ImportFrom` and :class:`ast.Attribute` node.
    Bare :class:`ast.Import` nodes (``import X``) are skipped — those
    bring a package into scope but do not reference a specific symbol;
    subsequent ``X.Y.Z`` usage is picked up by the attribute walk.

    Attribute chains that bottom out on anything other than a
    :class:`ast.Name` (a call result, a subscript) are dropped — those
    are dynamic and cannot be cross-referenced statically.

    Only *maximal* attribute chains are kept (a chain is dropped if any
    longer chain has it as a strict prefix) so a nested chain like
    ``molpy.builders.peptide.PeptideBuilder`` does not also surface as
    ``molpy.builders.peptide`` and ``molpy.builders``.
    """
    import_refs: set[str] = set()
    raw_chains: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not _starts_with_namespace(node.module):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                import_refs.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain is None or "." not in chain:
                continue
            root = chain.split(".", 1)[0]
            if root in MOLCRAFTS_NAMESPACES:
                raw_chains.add(chain)

    maximal = {
        c
        for c in raw_chains
        if not any(other != c and other.startswith(c + ".") for other in raw_chains)
    }
    return import_refs | maximal


def _starts_with_namespace(module: str) -> bool:
    head, _, _ = module.partition(".")
    return head in MOLCRAFTS_NAMESPACES


def _attribute_chain(node: ast.Attribute) -> str | None:
    """Reconstruct the full dotted name from an :class:`ast.Attribute` chain.

    Returns ``None`` if the chain bottoms out on anything other than a
    :class:`ast.Name` (e.g. a call result, a subscript).
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return ".".join(parts)
