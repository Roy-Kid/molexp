"""molexp **builtin** agent tools — always mounted, never MCP-discovered.

Hard-coded agent-facing tool **names** only. Surfaces:

* **chat** — inspect + code (scratch) + discover; no structure mutation / land
* **full** — chat tools plus ``workspace_ensure`` / ``run_land`` (archive path)
* **lifecycle** — full plus cancel/harvest (optional gate)

Third-party science APIs stay out of this list (auto-discovery law).
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.ops.protocols import ToolSpec

__all__ = [
    "ARCHIVE_TOOL_NAMES",
    "BUILTIN_SOURCE",
    "BUILTIN_TOOLS",
    "BUILTIN_TOOL_NAMES",
    "CHAT_TOOLS",
    "CHAT_TOOL_NAMES",
    "FULL_TOOL_NAMES",
    "OPS_TOOL_NAMES",
    "builtin_tool_specs",
    "lifecycle_builtin_specs",
    "tool_names_for_surface",
]

#: Wire / catalog source tag for Settings and ``discover`` hits.
BUILTIN_SOURCE = "builtin"


@dataclass(frozen=True, slots=True)
class BuiltinToolDef:
    """Static metadata for one agent tool."""

    name: str
    description: str
    #: (name, annotation, required) triples for the admin Tools list.
    parameters: tuple[tuple[str, str, bool], ...] = ()


# Chat surface — no authoritative workspace mutation.
CHAT_TOOLS: tuple[BuiltinToolDef, ...] = (
    BuiltinToolDef(
        name="workspace_inspect",
        description=(
            "Read-only: list a directory, or list projects / experiments under a project. "
            "Does not create folders or runs."
        ),
        parameters=(
            ("path", "str", False),
            ("project", "str | None", False),
        ),
    ),
    BuiltinToolDef(
        name="code_write",
        description=(
            "Write a UTF-8 file. In chat mode paths are confined under "
            "``agent/.scratch/`` (authoritative project/run trees are not writable)."
        ),
        parameters=(
            ("path", "str", True),
            ("content", "str", True),
        ),
    ),
    BuiltinToolDef(
        name="code_run",
        description=(
            "Run Python (cwd = workspace root). Provide exactly one of path= or code=. "
            "Chat scripts live under agent/.scratch/ — not under projects/…/runs/."
        ),
        parameters=(
            ("code", "str | None", False),
            ("path", "str | None", False),
            ("timeout", "float | None", False),
        ),
    ),
    BuiltinToolDef(
        name="embed_plot",
        description=(
            "Embed an interactive **molplot** chart in the conversation (Vega-Lite). "
            "Pass ``spec_json`` from ``molplot.line_spec`` / ``scatter_spec`` / "
            "``bar_spec`` (json.dumps). Prefer this over PNG / Markdown images."
        ),
        parameters=(
            ("title", "str", True),
            ("spec_json", "str", True),
        ),
    ),
    BuiltinToolDef(
        name="embed_structure",
        description=(
            "Embed a **molvis** structure viewer in the conversation. "
            "Provide ``content`` as XYZ/PDB/EXTXYZ text, or ``path`` under "
            "agent/.scratch/. Prefer this over dumping coordinates as prose."
        ),
        parameters=(
            ("title", "str", True),
            ("format", "str", True),
            ("content", "str | None", False),
            ("path", "str | None", False),
        ),
    ),
    BuiltinToolDef(
        name="discover",
        description=(
            "Search builtin tools, knowledge, and the live MCP catalog. "
            "Never invent third-party tool names."
        ),
        parameters=(
            ("query", "str", True),
            ("kind", "str | None", False),
        ),
    ),
    BuiltinToolDef(
        name="describe",
        description="Describe a discovery ref (builtin/MCP tool name or knowledge path).",
        parameters=(("ref", "str", True),),
    ),
)

# Archive / full surface — explicit land path (not default chat).
_ARCHIVE_TOOLS: tuple[BuiltinToolDef, ...] = (
    BuiltinToolDef(
        name="workspace_ensure",
        description=(
            "Create-or-get workspace structure (workspace | project | experiment | run). "
            "Idempotent. **Not mounted in chat mode** — use Plan, or full/archive surface "
            "when the user explicitly wants a durable Run."
        ),
        parameters=(
            ("kind", "str", True),
            ("name", "str", True),
            ("project", "str | None", False),
            ("experiment", "str | None", False),
            ("params_json", "str | None", False),
        ),
    ),
    BuiltinToolDef(
        name="run_land",
        description=(
            "Land products onto a Run and settle it (succeeded). **Not mounted in chat "
            "mode.** Only after a real run exists and products meet the MolRec/source "
            "standard — non-standard dumps must not land."
        ),
        parameters=(
            ("project", "str", True),
            ("experiment", "str", True),
            ("run_id", "str", True),
            ("files", "str | None", False),
            ("sources", "str | None", False),
            ("results_json", "str | None", False),
        ),
    ),
)

# Catalog order: archive structure → chat read/code/embed → land → discovery.
BUILTIN_TOOLS: tuple[BuiltinToolDef, ...] = (
    _ARCHIVE_TOOLS[0],  # workspace_ensure
    *CHAT_TOOLS[:3],  # inspect, code_write, code_run
    CHAT_TOOLS[3],  # embed_plot
    CHAT_TOOLS[4],  # embed_structure
    _ARCHIVE_TOOLS[1],  # run_land
    CHAT_TOOLS[5],  # discover
    CHAT_TOOLS[6],  # describe
)

CHAT_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in CHAT_TOOLS)
ARCHIVE_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in _ARCHIVE_TOOLS)
FULL_TOOL_NAMES: frozenset[str] = CHAT_TOOL_NAMES | ARCHIVE_TOOL_NAMES
BUILTIN_TOOL_NAMES: frozenset[str] = FULL_TOOL_NAMES

# Back-compat alias — full set (historical callers).
OPS_TOOL_NAMES = BUILTIN_TOOL_NAMES

# Optional builtins mounted only when InteractiveLoop operation_mode=lifecycle.
_LIFECYCLE_BUILTINS: tuple[BuiltinToolDef, ...] = (
    BuiltinToolDef(
        name="cancel_run",
        description="Cancel a live running run (workspace cancel verb).",
        parameters=(
            ("project_id", "str", True),
            ("experiment_id", "str", True),
            ("run_id", "str", True),
        ),
    ),
    BuiltinToolDef(
        name="harvest_run",
        description=(
            "Harvest a terminal run into a KnowledgeItem under its experiment. "
            "Fails softly on non-terminal runs."
        ),
        parameters=(
            ("project_id", "str", True),
            ("experiment_id", "str", True),
            ("run_id", "str", True),
            ("narrative", "str", True),
            ("kind", "str", False),
            ("created_by", "str", False),
        ),
    ),
)


def tool_names_for_surface(surface: str) -> frozenset[str]:
    """Return the builtin name set for ``chat`` or ``full``."""
    key = surface.strip().lower()
    if key in ("chat", "readonly", "default"):
        return CHAT_TOOL_NAMES
    if key in ("full", "archive", "ops"):
        return FULL_TOOL_NAMES
    if key == "lifecycle":
        return FULL_TOOL_NAMES  # lifecycle extras appended separately
    raise ValueError(f"unknown ops surface {surface!r}; use chat|full|lifecycle")


def builtin_tool_specs(*, surface: str = "full") -> tuple[ToolSpec, ...]:
    """Runtime :class:`ToolSpec` rows for the selected surface."""
    names = tool_names_for_surface(surface)
    return tuple(
        ToolSpec(name=t.name, description=t.description, source=BUILTIN_SOURCE)
        for t in BUILTIN_TOOLS
        if t.name in names
    )


def lifecycle_builtin_specs() -> tuple[ToolSpec, ...]:
    """Optional lifecycle builtins (cancel / harvest)."""
    return tuple(
        ToolSpec(name=t.name, description=t.description, source=f"{BUILTIN_SOURCE}:lifecycle")
        for t in _LIFECYCLE_BUILTINS
    )
