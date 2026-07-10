"""Stable tool adapter — few names, all protocol-backed.

Tool **names** here are the only hard-coded agent-facing identifiers.
They map to StructureOps / CodeEnv / Discovery — never to molpy symbols
or a frozen molmcp tool list.
"""

from __future__ import annotations

from collections.abc import Callable

from molexp.agent.ops.protocols import AgentSessionContext

OPS_TOOL_NAMES = frozenset(
    {
        "workspace_ensure",
        "workspace_inspect",
        "code_write",
        "code_run",
        "discover",
        "describe",
    }
)


def build_ops_tools(ctx: AgentSessionContext) -> tuple[Callable[..., str], ...]:
    """Return bare callables for pydantic-ai ``Agent(tools=...)``."""

    def workspace_ensure(
        kind: str,
        name: str,
        project: str | None = None,
    ) -> str:
        """Create-or-get workspace structure (idempotent).

        Args:
            kind: One of ``workspace``, ``project``, ``experiment``.
            name: Display name (slugified by molexp).
            project: Required when kind is ``experiment`` (project id or name).
        """
        try:
            k = kind.strip().lower()
            if k == "workspace":
                ref = ctx.structure.materialize(name=name or "workspace")
            elif k == "project":
                ref = ctx.structure.ensure_project(name)
            elif k == "experiment":
                if not project:
                    return "error: project= is required when kind=experiment"
                ref = ctx.structure.ensure_experiment(project, name)
            else:
                return f"error: unknown kind {kind!r}; use workspace|project|experiment"
            return f"ok kind={ref.kind} id={ref.id} path={ref.path}"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def workspace_inspect(path: str = ".", project: str | None = None) -> str:
        """List a directory, or list projects / experiments.

        Args:
            path: Workspace-relative path (default ``.``). Ignored when
                listing entities via empty path and project filter.
            project: If set, list experiments under this project instead
                of listing a directory.
        """
        try:
            if project is not None and (not path or path == "."):
                refs = ctx.structure.list_experiments(project)
                if not refs:
                    return f"(no experiments under project {project!r})"
                return "\n".join(f"{r.id}\t{r.name}\t{r.path}" for r in refs)
            if path in (".", "") and project is None:
                # Also surface top-level projects when inspecting root.
                view = ctx.structure.inspect(".")
                if view.error:
                    return f"error: {view.error}"
                lines = list(view.entries)
                projs = ctx.structure.list_projects()
                if projs:
                    lines.append("--- projects ---")
                    lines.extend(f"{p.id}\t{p.name}" for p in projs)
                return "\n".join(lines) if lines else "(empty)"
            view = ctx.structure.inspect(path)
            if view.error:
                return f"error: {view.error}"
            return "\n".join(view.entries) if view.entries else "(empty directory)"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def code_write(path: str, content: str) -> str:
        """Write a UTF-8 file under the workspace."""
        result = ctx.code.write(path, content)
        if not result.ok:
            return f"error: {result.error}"
        return f"wrote {result.path} ({result.bytes_written} bytes)"

    def code_run(
        code: str | None = None,
        path: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run Python under the workspace (cwd = workspace root).

        Provide exactly one of ``path`` (workspace script) or ``code``
        (inline). Use this for sweeps, harvest, molplot, and any import
        of installed packages — not a per-library tool.
        """
        result = ctx.code.run(path=path, code=code, timeout=timeout)
        if result.error and result.exit_code < 0:
            return f"error: {result.error}"
        return (
            f"exit_code={result.exit_code}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    def discover(query: str, kind: str | None = None) -> str:
        """Search knowledge and/or runtime MCP tool catalog.

        Args:
            query: Free text (tool name fragment, concept, keyword).
            kind: Optional ``mcp_tool`` or ``knowledge`` filter.
        """
        hits = ctx.discovery.search(query, kind=kind)
        if not hits:
            # Always also dump tool catalog summary when empty search.
            catalog = ctx.discovery.tools()
            if catalog and (kind in (None, "mcp_tool", "tool")):
                lines = ["(no knowledge hits; available MCP tools this session:)"]
                lines.extend(f"- {t.name}: {t.description[:120]}" for t in catalog[:40])
                return "\n".join(lines)
            return f"no matches for {query!r}"
        return "\n".join(f"{h.kind}\t{h.ref}\t{h.title}\t{h.summary}" for h in hits)

    def describe(ref: str) -> str:
        """Describe a discovery ref (knowledge path or MCP tool name)."""
        return ctx.discovery.describe(ref)

    tools: tuple[Callable[..., str], ...] = (
        workspace_ensure,
        workspace_inspect,
        code_write,
        code_run,
        discover,
        describe,
    )
    for t in tools:
        # ensure __name__ for protocol fakes / tests
        assert t.__name__ in OPS_TOOL_NAMES
    return tools


def render_discovery_catalog(ctx: AgentSessionContext) -> str:
    """Optional system appendix: live MCP tool names (not a static list)."""
    specs = ctx.discovery.tools()
    if not specs:
        return ""
    lines = ["## MCP tools available this session (runtime catalog)"]
    for s in specs[:60]:
        desc = s.description.replace("\n", " ")[:100]
        lines.append(f"- `{s.name}`: {desc}")
    if len(specs) > 60:
        lines.append(f"… ({len(specs) - 60} more; use discover)")
    return "\n".join(lines)
