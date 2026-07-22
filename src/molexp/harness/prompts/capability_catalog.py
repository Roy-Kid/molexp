"""Render a capability snapshot into a binder-facing catalog block.

``BindMolcraftsTasks`` appends this text (as a ``capability_catalog`` artifact,
threaded through ``AgentCallSpec.prompt_artifact_id``) to the binder's user
prompt when the run carries a grounded ``CapabilityRegistry``. It turns the
ungrounded "guess a capability_id" task into "pick one from this list", so the
binder's choices stay inside what ``ValidateBoundWorkflow`` will accept.

Noise (notes, tests, UI, non-science packages, private symbols, package/module
kinds) is dropped by :func:`molexp.harness.registry.bindable.is_bindable_capability`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.registry.bindable import is_bindable_capability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import CapabilitySelection, ToolCapability

__all__ = ["render_capability_catalog", "render_selected_capability_catalog"]

_MAX_DESC_CHARS = 120


def _is_bindable(cap: ToolCapability) -> bool:
    """Drop notes / tests / private / non-science entries (single gate)."""
    return is_bindable_capability(cap)


def _render_params(cap: ToolCapability) -> str:
    """Render ``(a, b*)`` from the capability's input schema (``*`` = required).

    When a molmcp routing guide is available, callers should prepend its
    checklist text so the binder sees package-role rules without molexp
    hard-coding package names.

    A schema without a ``properties`` key is a wildcard (unknown / ``**kwargs``
    signature) — rendered as ``(…)`` so the binder knows arguments are accepted
    without an enumerated set.
    """
    schema = cap.input_schema
    props = schema.get("properties")
    if not isinstance(props, dict):
        return "(…)"
    required = set(schema.get("required", []))
    return "(" + ", ".join(f"{name}*" if name in required else name for name in props) + ")"


def render_capability_catalog(capabilities: Iterable[ToolCapability]) -> str:
    """Render the bindable capabilities into a deterministic catalog block."""
    bindable = sorted((c for c in capabilities if _is_bindable(c)), key=lambda c: c.id)
    lines = [
        "## Available molcrafts capabilities",
        "",
        "Bind each PlanWorkflowIR task to exactly ONE `capability_id` chosen from "
        "this catalog. The `capability_id` is the dotted path **before** the "
        "opening `(` on each bullet — copy it VERBATIM. Do NOT invent ids from "
        "description prose, notes, or example names (e.g. never invent "
        "`molpy.build_polymer`). A trailing `*` marks a required parameter.",
        "",
    ]
    if not bindable:
        lines.append(
            "_No bindable science capabilities in the grounded registry "
            "(notes/tests/UI noise was filtered). Binding will fail validation "
            "if you invent an id — re-ground molmcp against molpy/molpack._"
        )
        return "\n".join(lines)
    for cap in bindable:
        desc = " ".join(cap.description.split())
        if len(desc) > _MAX_DESC_CHARS:
            desc = desc[: _MAX_DESC_CHARS - 1] + "…"
        # Never let description look like an alternate capability_id
        # (e.g. prose "molpy.build_polymer(...)").
        if desc.count("(") == 1 and desc.rstrip().endswith(")"):
            desc = ""
        suffix = f" — {desc}" if desc else ""
        lines.append(f"- {cap.id}{_render_params(cap)}{suffix}")
    return "\n".join(lines)


def render_selected_capability_catalog(
    registry: CapabilityRegistry, selection: CapabilitySelection
) -> str:
    """Render only the LLM-selected capabilities, each with its rationale.

    Unlike :func:`render_capability_catalog` (which dumps the whole grounded
    toolchain), this renders the narrowed set ``ResolveCapabilities`` chose for
    *this* experiment — one ``- id(params) — description`` line per capability
    followed by a ``↳ needed:`` line carrying the selector's reason. Selected
    ids the registry doesn't recognize are dropped (the registry is the source
    of truth for what binds) and listed in a trailing note.
    """
    lines = [
        "## Selected molcrafts capabilities",
        "",
        "These capabilities were chosen for THIS experiment from the grounded "
        "molcrafts toolchain. Bind each PlanWorkflowIR task to exactly ONE "
        "`capability_id` from this list (a trailing `*` marks a required "
        "parameter). Do NOT invent capability_ids that are not listed here.",
        "",
    ]
    dropped: list[str] = []
    for sel in selection.selected:
        if not registry.has(sel.id):
            dropped.append(sel.id)
            continue
        cap = registry.get(sel.id)
        desc = " ".join(cap.description.split())
        if len(desc) > _MAX_DESC_CHARS:
            desc = desc[: _MAX_DESC_CHARS - 1] + "…"
        suffix = f" — {desc}" if desc else ""
        lines.append(f"- {cap.id}{_render_params(cap)}{suffix}")
        reason = " ".join(sel.reason.split())
        if reason:
            lines.append(f"  ↳ needed: {reason}")
    if selection.notes.strip():
        lines += ["", f"_Note: {' '.join(selection.notes.split())}_"]
    if dropped:
        lines += ["", "_Dropped (not in the grounded toolchain): " + ", ".join(dropped) + "_"]
    return "\n".join(lines)
