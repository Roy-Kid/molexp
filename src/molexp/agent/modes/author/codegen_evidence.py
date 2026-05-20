"""Codegen-evidence gate — diff a generated module against the typed plan.

When AuthorMode generates a task implementation / test, the module may
reference a project symbol the plan never evidenced. :func:`validate_codegen_evidence`
walks the generated source with :mod:`ast` and returns one
:class:`MissingCapability` per tracked dotted reference that is *not*
backed by a node in the typed
:class:`~molexp.agent.modes._planning.CapabilityGraph`.

This is the re-typed successor of the old ``plan/capability.py`` AST
diff: the un-evidenced-symbol gate is now keyed off ``CapabilityGraph``
nodes' ``api_refs`` instead of a flat ``CapabilityEvidenceBatch``.

``ast_refs`` is the set of dotted paths reconstructed from
:class:`ast.ImportFrom` modules + maximal :class:`ast.Attribute` chains
whose leftmost :class:`ast.Name` matches a tracked namespace prefix.
``evidenced_refs`` is the union of every ``CapabilityNode.api_refs``.
A reference in ``ast_refs - evidenced_refs`` is a miss.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import CapabilityGraph

__all__ = [
    "FRAMEWORK_REFS",
    "MOLCRAFTS_NAMESPACES",
    "MissingCapability",
    "evidenced_refs",
    "extract_ast_refs",
    "validate_codegen_evidence",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

MOLCRAFTS_NAMESPACES: tuple[str, ...] = ("molexp", "molpy", "molq", "molvis", "molrs")
"""Top-level project namespaces whose dotted references the codegen gate
tracks. A reference outside these prefixes (stdlib, third-party) is not
gated — only project-specific symbols require capability evidence."""

FRAMEWORK_REFS: tuple[str, ...] = ("molexp.workflow",)
"""Codegen-framework references generated modules always need (the
``molexp.workflow.Task`` base class, ``TaskContext``, …). They are the
scaffolding AuthorMode emits — not a project capability the plan must
evidence — so the gate exempts every reference under these prefixes."""


class MissingCapability(BaseModel):
    """One un-evidenced symbol the codegen-evidence diff turned up.

    Attributes:
        ref: The dotted reference the generated module used.
        reason: Why the reference is flagged (always
            ``unevidenced_in_code`` for this gate).
        detail: A human-readable description.
    """

    model_config = _FROZEN

    ref: str
    reason: str = "unevidenced_in_code"
    detail: str = ""


def evidenced_refs(capability_graph: CapabilityGraph) -> frozenset[str]:
    """Return every ``api_ref`` backed by a :class:`CapabilityGraph` node."""
    refs: set[str] = set()
    for node in capability_graph.nodes:
        refs.update(node.api_refs)
    return frozenset(refs)


def _tracked_namespaces(capability_graph: CapabilityGraph) -> frozenset[str]:
    """Return the top-level namespaces the diff tracks for this plan.

    The base :data:`MOLCRAFTS_NAMESPACES` plus the leading segment of
    every evidenced ``api_ref`` (so a plan that legitimately uses a
    namespace outside the default list still gets gated coverage).
    """
    tracked = set(MOLCRAFTS_NAMESPACES)
    for ref in evidenced_refs(capability_graph):
        head, _, _ = ref.partition(".")
        if head:
            tracked.add(head)
    return frozenset(tracked)


def _attribute_chain(node: ast.Attribute) -> str | None:
    """Reconstruct the full dotted name from an :class:`ast.Attribute` chain.

    Returns ``None`` when the chain bottoms out on anything other than a
    :class:`ast.Name` (a call result, a subscript) — those are dynamic
    and cannot be cross-referenced statically.
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


def extract_ast_refs(tree: ast.Module, *, namespaces: Iterable[str]) -> frozenset[str]:
    """Reconstruct tracked dotted paths from imports + attribute chains.

    Walks every :class:`ast.ImportFrom` and :class:`ast.Attribute` node;
    keeps only *maximal* attribute chains so nested chains do not also
    surface their intermediate module prefixes.
    """
    tracked = frozenset(namespaces)
    import_refs: set[str] = set()
    raw_chains: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head, _, _ = node.module.partition(".")
            if head not in tracked:
                continue
            for alias in node.names:
                if alias.name != "*":
                    import_refs.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain is None or "." not in chain:
                continue
            if chain.split(".", 1)[0] in tracked:
                raw_chains.add(chain)
    maximal = {
        chain
        for chain in raw_chains
        if not any(other != chain and other.startswith(chain + ".") for other in raw_chains)
    }
    return frozenset(import_refs | maximal)


def validate_codegen_evidence(
    source: str,
    capability_graph: CapabilityGraph,
) -> tuple[MissingCapability, ...]:
    """Diff a generated module's project references against the plan's evidence.

    Returns one :class:`MissingCapability` per tracked dotted reference
    in ``source`` that is absent from the typed
    :class:`CapabilityGraph`. An empty tuple means every project symbol
    the module uses is evidenced.

    Raises:
        SyntaxError: if ``source`` does not parse.
    """
    tree = ast.parse(source)
    tracked = _tracked_namespaces(capability_graph)
    refs = extract_ast_refs(tree, namespaces=tracked)
    backed = evidenced_refs(capability_graph)
    misses = [
        MissingCapability(
            ref=ref,
            detail=f"{ref} is not backed by any CapabilityGraph node",
        )
        for ref in sorted(refs - backed)
        if not _is_framework_ref(ref)
    ]
    return tuple(misses)


def _is_framework_ref(ref: str) -> bool:
    """Return whether ``ref`` is codegen-framework scaffolding (always allowed)."""
    return any(ref == prefix or ref.startswith(prefix + ".") for prefix in FRAMEWORK_REFS)
