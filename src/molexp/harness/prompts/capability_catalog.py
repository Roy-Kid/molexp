"""Render a capability snapshot into a binder-facing catalog block.

``BindMolcraftsTasks`` appends this text (as a ``capability_catalog`` artifact,
threaded through ``AgentCallSpec.prompt_artifact_id``) to the binder's user
prompt when the run carries a grounded ``CapabilityRegistry``. It turns the
ungrounded "guess a capability_id" task into "pick one from this list", so the
binder's choices stay inside what ``ValidateBoundWorkflow`` will accept.

Private (``_``-prefixed) symbols and module-kind entries are dropped — they are
real index hits but not capabilities a workflow should bind to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import CapabilitySelection, ToolCapability

__all__ = ["render_capability_catalog", "render_selected_capability_catalog"]

_MAX_DESC_CHARS = 120


def _is_bindable(cap: ToolCapability) -> bool:
    """Drop module-kind entries and private (``_``-prefixed leaf) symbols."""
    if cap.tags and cap.tags[0] == "module":
        return False
    return not cap.id.rsplit(".", 1)[-1].startswith("_")


def _render_params(cap: ToolCapability) -> str:
    """Render ``(a, b*)`` from the capability's input schema (``*`` = required).

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
        "Bind each WorkflowIR task to exactly ONE `capability_id` chosen from "
        "this catalog. Do NOT invent capability_ids, callables, or packages that "
        "are not listed here. A trailing `*` marks a required parameter.",
        "",
    ]
    for cap in bindable:
        desc = " ".join(cap.description.split())
        if len(desc) > _MAX_DESC_CHARS:
            desc = desc[: _MAX_DESC_CHARS - 1] + "…"
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
        "molcrafts toolchain. Bind each WorkflowIR task to exactly ONE "
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
