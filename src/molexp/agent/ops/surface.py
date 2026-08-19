"""Declared tool surface — the coeffect spec a loop/mode provides.

A :class:`ToolSurface` is the set of :class:`SurfaceKey`\\ s a turn may
satisfy. Builtins declare their required keys; unknown MCP tools have
an empty requirement unless the **classifier** tags them as
``workspace_mutate`` (name needles, never a per-MCP-tool table).

Policy is subset, not a deny-list: a tool runs iff
``required ⊆ surface.keys``. No shipped preset provides
``workspace_mutate`` — those names stay a closed molexp effect class.
``workspace_ensure`` / ``run_land`` are *declared* ``archive`` tools
and are not classified as mutators.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "CHAT_SURFACE",
    "FULL_SURFACE",
    "LIFECYCLE_SURFACE",
    "WORKSPACE_MUTATE_NEEDLES",
    "SurfaceKey",
    "ToolSurface",
    "classify_undeclared",
    "deny_reason",
    "is_workspace_mutator_name",
    "required_keys",
    "surface_for_mode",
]


class SurfaceKey(StrEnum):
    """Capability a surface may provide. Tools declare what they require."""

    INSPECT = "inspect"
    SCRATCH_FS = "scratch_fs"
    EMBED = "embed"
    DISCOVERY = "discovery"
    ARCHIVE = "archive"
    LIFECYCLE = "lifecycle"
    WORKSPACE_MUTATE = "workspace_mutate"


@dataclass(frozen=True, slots=True)
class ToolSurface:
    """Named set of keys a loop turn provides.

    Attributes:
        name: Preset id (``chat`` / ``full`` / ``lifecycle``).
        keys: Keys this surface satisfies.
    """

    name: str
    keys: frozenset[SurfaceKey]

    def provides(self, key: SurfaceKey) -> bool:
        """Return whether this surface satisfies *key*."""
        return key in self.keys

    def allows(self, required: frozenset[SurfaceKey]) -> bool:
        """Return whether ``required ⊆ self.keys``."""
        return required <= self.keys


CHAT_SURFACE = ToolSurface(
    name="chat",
    keys=frozenset(
        {
            SurfaceKey.INSPECT,
            SurfaceKey.SCRATCH_FS,
            SurfaceKey.EMBED,
            SurfaceKey.DISCOVERY,
        }
    ),
)
FULL_SURFACE = ToolSurface(
    name="full",
    keys=CHAT_SURFACE.keys | {SurfaceKey.ARCHIVE},
)
LIFECYCLE_SURFACE = ToolSurface(
    name="lifecycle",
    keys=FULL_SURFACE.keys | {SurfaceKey.LIFECYCLE},
)

#: MCP-only classifier needles. Builtin archive names are **not** listed —
#: their :class:`~molexp.agent.ops.builtins.BuiltinToolDef.required` is
#: authoritative. Matching is case-insensitive substring on the tool name.
WORKSPACE_MUTATE_NEEDLES: tuple[str, ...] = (
    "add_project",
    "add_experiment",
    "create_run",
    "remove_project",
    "remove_experiment",
    "delete_project",
    "delete_experiment",
    "delete_run",
    "molexp_materialize",
)


def surface_for_mode(operation_mode: str) -> ToolSurface:
    """Map ``operation_mode`` onto a preset tool surface.

    Args:
        operation_mode: ``chat`` / ``full`` / ``lifecycle``.

    Returns:
        The matching preset.

    Raises:
        ValueError: Unknown mode string.
    """
    key = operation_mode.strip().lower()
    if key == "chat":
        return CHAT_SURFACE
    if key == "full":
        return FULL_SURFACE
    if key == "lifecycle":
        return LIFECYCLE_SURFACE
    raise ValueError(f"unknown ops surface {operation_mode!r}; use chat|full|lifecycle")


def is_workspace_mutator_name(tool_name: str) -> bool:
    """Return True if *tool_name* looks like an undeclared workspace mutator."""
    low = tool_name.lower().replace("-", "_")
    return any(n in low for n in WORKSPACE_MUTATE_NEEDLES)


def classify_undeclared(tool_name: str) -> frozenset[SurfaceKey]:
    """Requirement for a name with no builtin declaration (live MCP).

    Empty unless the classifier tags ``workspace_mutate``. Never grows a
    per-MCP-tool table (auto-discovery law).
    """
    if is_workspace_mutator_name(tool_name):
        return frozenset({SurfaceKey.WORKSPACE_MUTATE})
    return frozenset()


def required_keys(
    tool_name: str,
    *,
    declared: Mapping[str, frozenset[SurfaceKey]] | None = None,
) -> frozenset[SurfaceKey]:
    """Resolve the required keys for *tool_name*.

    Declared builtins win. Everything else goes through
    :func:`classify_undeclared`.
    """
    if declared is not None and tool_name in declared:
        return declared[tool_name]
    return classify_undeclared(tool_name)


def deny_reason(
    surface: ToolSurface,
    tool_name: str,
    *,
    declared: Mapping[str, frozenset[SurfaceKey]] | None = None,
) -> str | None:
    """Return a deny message if *tool_name* is not allowed on *surface*.

    ``None`` means the tool may run. The loop turns this string into a
    :class:`~molexp.agent.loops.hooks.HookOutcome` so ``ops`` does not
    import the hook vocabulary.
    """
    needed = required_keys(tool_name, declared=declared)
    if surface.allows(needed):
        return None
    missing = ", ".join(sorted(k.value for k in (needed - surface.keys)))
    return (
        f"{surface.name} surface does not provide {missing} "
        f"(tool {tool_name!r}). Authoritative workspace mutation "
        "is not on this surface — keep work under agent/.scratch/ "
        "or use a surface that provides the missing key."
    )
