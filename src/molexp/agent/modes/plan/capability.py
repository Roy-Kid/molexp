"""Capability discovery schemas + codegen-evidence validator.

Owns the five frozen pydantic models the PlanMode capability-discovery
pipeline uses plus a pure-AST validator that codegen nodes invoke
after writing each generated module to disk.

PlanMode deals only in structured needs, hints, evidence, and
validation misses. Source-specific discovery policy and transport
selection live behind the agent-level discovery service.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from molexp.agent.capability_hints import (
    LEGACY_VALIDATION_NAMESPACES as MOLCRAFTS_NAMESPACES,
)
from molexp.agent.capability_hints import (
    CapabilityHint,
    validate_hint_constraints,
)

__all__ = [
    "MOLCRAFTS_NAMESPACES",
    "CapabilityEvidence",
    "CapabilityEvidenceBatch",
    "CapabilityHint",
    "CapabilityNeed",
    "CapabilityNeedReport",
    "MissReason",
    "MissingCapability",
    "extract_ast_refs",
    "extract_declared_refs",
    "validate_codegen_evidence",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


MissReason = Literal[
    "mcp_no_match",
    "mcp_low_confidence",
    "mcp_timeout",
    "unevidenced_in_code",
    "undeclared_in_code",
    "declared_but_unused",
    "required_namespace_missing",
    "required_namespace_unused",
    "fallback_reason_missing",
    "forbidden_hand_rolled_output",
]
"""Closed enumeration of valid :attr:`MissingCapability.reason` values."""


# ── Frozen pydantic models ─────────────────────────────────────────────────


class CapabilityNeed(BaseModel):
    """One unit of capability needed by an experiment task.

    Drafted by ``DraftCapabilityNeeds`` from the plan-brief stages and
    persisted under ``capability/needs.yaml``.

    Attributes:
        task_id: Opaque stable identifier the need attaches to.
            Discovery runs *before* the workflow IR is compiled, so at
            draft time this is a free-form stage label drawn from
            ``PlanBrief.stages`` (or any identifier the LLM can echo
            consistently — ``"stage:0"``, ``"build_system"``, …). When
            the IR compiler later emits the matching task, the
            ``TaskIRBrief.task_id`` is expected to reuse the same
            string; the validator does not assert on this beyond
            evidence-batch lookups keyed by ``api_ref``.
        capability: Natural-language description (one phrase) of what
            the step needs — e.g. ``"construct a peptide from amino-acid codes"``.
        rationale: One sentence saying *why* the capability is needed
            for the step to succeed.
        expected_kind: Hint about what shape the symbol should take.
            Canonical strings: ``"class"`` / ``"callable"`` / ``"module"`` /
            ``"constant"`` / ``"protocol"`` / ``"namespace"``. Unknown
            values are not enforced at the type level (different
            sources may surface other kinds).
        query_hints: Optional query keywords passed to the discovery
            source to bias the result.
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
            the task is pure-stdlib). ``True`` forces concrete discovery.
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
    hints: tuple[CapabilityHint, ...] = ()


class CapabilityEvidence(BaseModel):
    """Resolved evidence for one need.

    Attributes:
        need_fingerprint: Stable identifier of the originating
            :class:`CapabilityNeed` — typically ``f"{task_id}:{capability}"``.
            Lets discovery results match back to their need without
            embedding the full need object.
        source: Discovery channel that produced this evidence.
        namespace: Top-level namespace this evidence belongs to.
        package: Compatibility package field for older evidence rows.
        module: Fully-qualified module path.
        symbol: Symbol name within the module.
        kind: Symbol kind (``"class"`` / ``"callable"`` / etc.).
        signature: Canonical Python signature line.
        doc_summary: First-paragraph summary of the symbol's docstring.
        api_ref: Canonical Python dotted-path identifier, equivalent to
            ``f"{module}.{symbol}"``. **Primary key for the
            codegen-evidence diff** in :func:`validate_codegen_evidence`.
        confidence: ``0.0..1.0`` confidence score returned by the
            discovery source.
    """

    model_config = _FROZEN

    need_fingerprint: str
    source: str
    namespace: str = ""
    package: str
    module: str
    symbol: str
    kind: str
    signature: str
    doc_summary: str = ""
    usage_notes: tuple[str, ...] = ()
    api_ref: str
    confidence: float = 1.0


class MissingCapability(BaseModel):
    """One row in the missing-capability ledger.

    Produced by both the discovery service (for source miss reasons) and by
    :func:`validate_codegen_evidence` (for ``unevidenced_in_code`` /
    ``undeclared_in_code`` / ``declared_but_unused``).

    Attributes:
        need: Originating :class:`CapabilityNeed`. ``None`` for
            validator-emitted misses where no upstream need maps to the
            offending ref (the diff is keyed on ``api_ref`` only).
        reason: One of the values defined by :data:`MissReason`;
            constrained at the type level.
        detail: Human-readable description; for validator-emitted misses
            includes the offending ``api_ref`` so the repair loop can
            re-target discovery.
        repairable: Whether the repair loop should retry. ``False``
            signals a permanent miss.
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
        missing: Needs that could not be evidenced.
        discovery_skipped: Set when ``discovery_required`` was ``False``
            upstream; tells codegen + validator to relax the
            ``__capability_evidence__`` block requirement entirely.
    """

    model_config = _FROZEN

    evidence: tuple[CapabilityEvidence, ...] = ()
    missing: tuple[MissingCapability, ...] = ()
    discovery_skipped: bool = False
    hints: tuple[CapabilityHint, ...] = ()
    tracked_namespaces: tuple[str, ...] = ()
    fallback_reasons: tuple[str, ...] = ()


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
    declared_refs = extract_declared_refs(tree)
    tracked_namespaces = _tracked_namespaces(batch)
    ast_refs = extract_ast_refs(tree, namespaces=tracked_namespaces)
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

    used_refs = ast_refs | declared_refs
    for hint in batch.hints:
        if hint.strength == "required" and not _namespace_is_used(hint.namespace, used_refs):
            misses.append(
                MissingCapability(
                    need=None,
                    reason="required_namespace_unused",
                    detail=f"required namespace {hint.namespace} was not used by generated code",
                    repairable=True,
                )
            )
        if (
            hint.strength == "preferred"
            and not _namespace_is_used(hint.namespace, used_refs | evidence_refs)
            and not any(hint.namespace in reason for reason in batch.fallback_reasons)
        ):
            misses.append(
                MissingCapability(
                    need=None,
                    reason="fallback_reason_missing",
                    detail=(
                        f"preferred namespace {hint.namespace} was not used and no "
                        "fallback reason was recorded"
                    ),
                    repairable=True,
                )
            )

    for violation in validate_hint_constraints(source, batch.hints):
        misses.append(
            MissingCapability(
                need=None,
                reason=violation.reason,  # type: ignore[arg-type]
                detail=violation.detail,
                repairable=True,
            )
        )

    return tuple(misses)


def extract_declared_refs(tree: ast.Module) -> set[str]:
    """Pull ``__capability_evidence__: tuple[str, ...] = (...)`` literals.

    Handles both the annotated (:class:`ast.AnnAssign`) and plain
    (:class:`ast.Assign`) forms; only module-level assignments count.
    Returns the empty set when no such assignment exists or the value
    is not a string-tuple literal.

    Public so the codegen-gate helpers in :mod:`tasks` can re-use the
    same extraction without reaching across the privacy fence.
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


def extract_ast_refs(
    tree: ast.Module,
    *,
    namespaces: Iterable[str] | None = None,
) -> set[str]:
    """Reconstruct tracked dotted paths from imports + attribute chains.

    Walks every :class:`ast.ImportFrom` and :class:`ast.Attribute` node.
    Bare :class:`ast.Import` nodes (``import X``) are skipped — those
    bring a package into scope but do not reference a specific symbol;
    subsequent ``X.Y.Z`` usage is picked up by the attribute walk.

    Attribute chains that bottom out on anything other than a
    :class:`ast.Name` (a call result, a subscript) are dropped — those
    are dynamic and cannot be cross-referenced statically.

    Only *maximal* attribute chains are kept (a chain is dropped if any
    longer chain has it as a strict prefix) so nested chains do not
    also surface their intermediate module prefixes.
    """
    tracked = frozenset(namespaces if namespaces is not None else MOLCRAFTS_NAMESPACES)
    import_refs: set[str] = set()
    raw_chains: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not _starts_with_namespace(node.module, tracked):
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
            if root in tracked:
                raw_chains.add(chain)

    maximal = {
        c
        for c in raw_chains
        if not any(other != c and other.startswith(c + ".") for other in raw_chains)
    }
    return import_refs | maximal


def _starts_with_namespace(module: str, namespaces: Iterable[str]) -> bool:
    head, _, _ = module.partition(".")
    return head in namespaces


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


def _tracked_namespaces(batch: CapabilityEvidenceBatch) -> tuple[str, ...]:
    tracked = set(batch.tracked_namespaces)
    tracked.update(hint.namespace for hint in batch.hints)
    for evidence in batch.evidence:
        namespace = evidence.namespace or evidence.package
        if namespace:
            tracked.add(namespace)
    if not tracked:
        tracked.update(MOLCRAFTS_NAMESPACES)
    return tuple(sorted(tracked))


def _namespace_is_used(namespace: str, refs: Iterable[str]) -> bool:
    prefix = f"{namespace}."
    return any(ref == namespace or ref.startswith(prefix) for ref in refs)
