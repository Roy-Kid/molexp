"""Stable agent-ops protocols — extensible without hardcoding upstream APIs.

Implementations may call workspace, ExecutionEnv, or open MCP toolsets.
They must **never** embed a hand-curated table of molpy/molpack/… symbols
or molmcp tool names that must be updated when those repos change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WriteResult:
    """Outcome of :meth:`CodeEnv.write`."""

    path: str
    bytes_written: int
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Outcome of :meth:`CodeEnv.run`."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class EntityRef:
    """Lightweight identity for a workspace entity."""

    kind: str  # workspace | project | experiment | run | path
    id: str
    path: str = ""
    name: str = ""


@dataclass(frozen=True)
class TreeView:
    """Directory listing / inspect payload."""

    path: str
    entries: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class Hit:
    """One discovery hit — ref is opaque across packages."""

    ref: str
    kind: str = ""  # mcp_tool | knowledge | …
    title: str = ""
    summary: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class ToolSpec:
    """Runtime-discovered tool (e.g. from an open MCP session)."""

    name: str
    description: str = ""
    source: str = ""


@runtime_checkable
class CodeEnv(Protocol):
    """Write files and run Python under the session workspace.

    The **only** channel for science / plotting: agent code imports
    molexp, molplot, molpy, … as ordinary packages.
    """

    def write(self, path: str, content: str) -> WriteResult:
        """Write UTF-8 text under the workspace root."""
        ...

    def run(
        self,
        *,
        path: str | None = None,
        code: str | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        """Execute a workspace script or inline snippet (cwd = workspace)."""
        ...


@runtime_checkable
class StructureOps(Protocol):
    """Idempotent workspace structure — molexp Folder family only.

    Not a science executor: no sweep, no workflow runtime drive, no plot.
    """

    def materialize(self, name: str = "workspace") -> EntityRef:
        """Ensure the workspace root exists on disk."""
        ...

    def ensure_project(self, name: str) -> EntityRef:
        """Create-or-get a project (idempotent on slug)."""
        ...

    def ensure_experiment(self, project: str, name: str) -> EntityRef:
        """Create-or-get an experiment under *project* (idempotent on slug)."""
        ...

    def inspect(self, path: str = ".") -> TreeView:
        """List a workspace-relative (or under-root absolute) path."""
        ...

    def list_projects(self) -> tuple[EntityRef, ...]:
        """List projects under the workspace."""
        ...

    def list_experiments(self, project: str) -> tuple[EntityRef, ...]:
        """List experiments under *project*."""
        ...


@runtime_checkable
class Discovery(Protocol):
    """Auto-discovery surface — catalogs are runtime-enumerated only.

    Implementations query MCP tool lists, knowledge bundles, or molmcp
    search. They must not ship a static table of third-party APIs.
    """

    def tools(self) -> tuple[ToolSpec, ...]:
        """Currently available external tools (MCP, …) for this session."""
        ...

    def search(self, query: str, *, kind: str | None = None) -> tuple[Hit, ...]:
        """Search knowledge / catalog by free text."""
        ...

    def describe(self, ref: str) -> str:
        """Full detail for one hit ref (knowledge path or tool name)."""
        ...


@runtime_checkable
class BehaviorPolicy(Protocol):
    """Mode-like behavior only — never a capability mask."""

    def system_preamble(self) -> str:
        """Text composed into the system prompt before user overrides."""
        ...


@dataclass(frozen=True)
class AgentSessionContext:
    """Per-turn handle: workspace + three ops protocols + behavior.

    External capability **names** are never stored as constants — only
    the :class:`Discovery` protocol that enumerates them at runtime.
    """

    workspace_root: Path
    code: CodeEnv
    structure: StructureOps
    discovery: Discovery
    behavior: BehaviorPolicy
