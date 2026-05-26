"""Codegen-evidence gate — diff a generated module against the typed plan.

When AuthorMode generates a task implementation / test, the module may
reference a project symbol the plan never grounded. :func:`validate_codegen_evidence`
walks the generated source with :mod:`ast` and returns one
:class:`MissingCapability` per tracked dotted reference that is *not* in
the union of every :class:`~molexp.agent.modes._planning.PlanStep`'s
``api_refs``.

Capability evidence lives inline on each :class:`PlanStep` (``api_refs``
+ ``composition_notes``). This module's gate keys off the union of
``step.api_refs`` across the plan — no separate graph artefact.

``ast_refs`` is the set of dotted paths reconstructed from
:class:`ast.ImportFrom` modules + maximal :class:`ast.Attribute` chains
whose leftmost :class:`ast.Name` matches a tracked namespace prefix.
A reference in ``ast_refs - evidenced_refs`` is a miss.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import PlanGraph

__all__ = [
    "FRAMEWORK_REFS",
    "MissingCapability",
    "evidenced_refs",
    "extract_ast_refs",
    "validate_codegen_evidence",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

FRAMEWORK_REFS: tuple[str, ...] = ("molexp.workflow",)
"""Codegen-framework references generated modules always need (the
``molexp.workflow.Task`` base class, ``TaskContext``, …). They are the
scaffolding AuthorMode emits — not a project capability the plan must
evidence — so the gate exempts every reference under these prefixes.
``molexp.workflow`` is the *only* hardcoded namespace: it is the
codegen-side framework contract, not a project the planner discovered."""


class MissingCapability(BaseModel):
    """One un-evidenced symbol the codegen-evidence diff turned up."""

    model_config = _FROZEN

    ref: str
    reason: str = "unevidenced_in_code"
    detail: str = ""


def evidenced_refs(plan_graph: PlanGraph) -> frozenset[str]:
    """Return every ``api_ref`` named by any :class:`PlanStep`.

    The plan carries its api_refs inline on each step; the evidence set
    is the union of ``step.api_refs`` across the plan.
    """
    refs: set[str] = set()
    for step in plan_graph.steps:
        refs.update(step.api_refs)
    return frozenset(refs)


def _tracked_namespaces(plan_graph: PlanGraph) -> frozenset[str]:
    """Return the top-level namespaces the diff tracks for this plan.

    Derived purely from the leading segment of every ``api_ref`` the
    planner discovered — no hardcoded package list. A plan that
    legitimately uses any namespace gets gated coverage as long as
    that namespace appears in at least one ``PlanStep.api_refs`` entry.
    """
    tracked: set[str] = set()
    for ref in evidenced_refs(plan_graph):
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
    """Reconstruct tracked dotted paths from imports + attribute chains."""
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
    plan_graph: PlanGraph,
) -> tuple[MissingCapability, ...]:
    """Diff a generated module's project references against the plan's evidence.

    Returns one :class:`MissingCapability` per tracked dotted reference
    in ``source`` that is absent from the union of every
    :class:`PlanStep`'s ``api_refs``. An empty tuple means every
    project symbol the module uses is grounded somewhere in the plan.

    Re-exports are tolerated: a generated ``molpy.Atomistic`` is
    accepted when some api_ref ends in ``.Atomistic`` and shares the
    same top-level namespace as some api_ref. This matches Python's actual
    import semantics — a package's ``__init__.py`` typically re-exports
    canonical symbols at the top level, and the gate's job is to catch
    *invented* symbols, not to demand a specific dotted path.

    Raises:
        SyntaxError: if ``source`` does not parse.
    """
    tree = ast.parse(source)
    tracked = _tracked_namespaces(plan_graph)
    refs = extract_ast_refs(tree, namespaces=tracked)
    backed = evidenced_refs(plan_graph)
    misses = [
        MissingCapability(
            ref=ref,
            detail=f"{ref} is not backed by any PlanStep.api_refs entry",
        )
        for ref in sorted(refs)
        if not _is_framework_ref(ref) and not _is_backed(ref, backed)
    ]
    return tuple(misses)


def _is_backed(ref: str, backed: frozenset[str]) -> bool:
    """Return whether ``ref`` is grounded by any entry in ``backed``.

    Three acceptance modes:

    1. **Exact**: ``ref`` is literally in ``backed``.
    2. **Re-export by symbol-name match**: ``ref`` shares its top-level
       namespace with some backed entry AND ends with the same final
       segment (the symbol name). ``molpy.Atomistic`` is backed by
       ``molpy.core.atomistic.Atomistic`` (top + symbol both match) and
       vice-versa.
    3. **Container of a backed method**: ``ref`` is the class / module
       qualname that some backed entry lives under, i.e. some backed
       entry starts with ``ref + "."``. ``molpy.core.atomistic.Atomistic``
       is implicitly backed when api_refs contains
       ``molpy.core.atomistic.Atomistic.to_frame`` — you cannot call the
       method without importing the class.
    """
    if ref in backed:
        return True
    if "." not in ref:
        return False
    prefix = f"{ref}."
    if any(b.startswith(prefix) for b in backed):
        return True
    top, _, _ = ref.partition(".")
    symbol = ref.rsplit(".", 1)[-1]
    suffix = f".{symbol}"
    namespace = f"{top}."
    return any(b.startswith(namespace) and b.endswith(suffix) for b in backed)


def _is_framework_ref(ref: str) -> bool:
    """Return whether ``ref`` is codegen-framework scaffolding (always allowed)."""
    return any(ref == prefix or ref.startswith(prefix + ".") for prefix in FRAMEWORK_REFS)
