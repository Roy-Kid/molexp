"""Builtin tool adapter — molexp-owned tools mounted on InteractiveLoop.

Tool **names** here are the only hard-coded agent-facing identifiers.
Surfaces: ``chat`` (default) vs ``full`` (archive: ensure + land).
"""

from __future__ import annotations

from collections.abc import Callable

from molexp.agent.ops.builtins import (
    BUILTIN_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    FULL_TOOL_NAMES,
    OPS_TOOL_NAMES,
    builtin_tool_specs,
    tool_names_for_surface,
)
from molexp.agent.ops.protocols import AgentSessionContext, ToolSpec

__all__ = [
    "BUILTIN_TOOL_NAMES",
    "CHAT_TOOL_NAMES",
    "FULL_TOOL_NAMES",
    "OPS_TOOL_NAMES",
    "build_ops_tools",
    "render_discovery_catalog",
]


def build_ops_tools(
    ctx: AgentSessionContext,
    *,
    surface: str = "chat",
) -> tuple[Callable[..., str], ...]:
    """Return builtin callables for pydantic-ai ``Agent(tools=...)``.

    Args:
        ctx: Session ops context.
        surface: ``chat`` (default — no ensure/land) or ``full`` / ``lifecycle``
            (includes archive tools).
    """
    allowed = tool_names_for_surface(surface)

    def workspace_ensure(
        kind: str,
        name: str,
        project: str | None = None,
        experiment: str | None = None,
        params_json: str | None = None,
    ) -> str:
        """Create-or-get workspace structure (idempotent). Full surface only."""
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
            elif k == "run":
                if not project or not experiment:
                    return "error: project= and experiment= are required when kind=run"
                params: dict[str, object] | None = None
                if params_json and params_json.strip():
                    import json

                    raw = json.loads(params_json)
                    if not isinstance(raw, dict):
                        return "error: params_json must be a JSON object"
                    params = dict(raw)
                    if name and "label" not in params:
                        params["label"] = name
                elif name:
                    params = {"label": name}
                ref = ctx.structure.ensure_run(project, experiment, params=params)
            else:
                return f"error: unknown kind {kind!r}; use workspace|project|experiment|run"
            return f"ok kind={ref.kind} id={ref.id} path={ref.path}"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def workspace_inspect(path: str = ".", project: str | None = None) -> str:
        """List a directory, or list projects / experiments (read-only)."""
        try:
            if project is not None and (not path or path == "."):
                refs = ctx.structure.list_experiments(project)
                if not refs:
                    return f"(no experiments under project {project!r})"
                return "\n".join(f"{r.id}\t{r.name}\t{r.path}" for r in refs)
            if path in (".", "") and project is None:
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
        """Write a UTF-8 file (chat: under agent/.scratch/)."""
        result = ctx.code.write(path, content)
        if not result.ok:
            return f"error: {result.error}"
        return f"wrote {result.path} ({result.bytes_written} bytes)"

    def code_run(
        code: str | None = None,
        path: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run Python (chat: cwd under agent/.scratch/; full: workspace root)."""
        result = ctx.code.run(path=path, code=code, timeout=timeout)
        if result.error and result.exit_code < 0:
            return f"error: {result.error}"
        out = (
            f"exit_code={result.exit_code}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        if result.exit_code == 0:
            confine = getattr(ctx.code, "_confine", None)
            if confine:
                out += (
                    "\n--- chat hint ---\n"
                    "cwd is agent/.scratch/ only. **Do not** call Workspace."
                    "add_project / add_experiment / add_run or write under projects/ — "
                    "Chat never creates structure. "
                    "Show charts via **embed_plot** (molplot VL JSON), structures via "
                    "**embed_structure**. Then ask (in English) whether to archive/land."
                )
            else:
                out += (
                    "\n--- chat hint ---\n"
                    "To show a chart in the conversation call **embed_plot** "
                    "(molplot Vega-Lite via molplot.line_spec/scatter_spec + json.dumps). "
                    "To show a molecule call **embed_structure**. "
                    "Do not Markdown-embed PNG. Then ask (in English) whether to "
                    "archive/land onto a formal experiment/run."
                )
        return out

    def embed_plot(title: str, spec_json: str) -> str:
        """Embed a molplot (Vega-Lite) chart in the conversation UI."""
        import json

        from molexp.agent.ops.embed import encode_embed_result

        try:
            spec = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            return f"error: spec_json is not valid JSON: {exc}"
        if not isinstance(spec, dict):
            return "error: spec_json must be a JSON object (Vega-Lite spec)"
        # Accept either raw VL or {spec: VL} wrappers.
        if "spec" in spec and isinstance(spec["spec"], dict) and "mark" not in spec:
            payload = dict(spec["spec"])
        else:
            payload = dict(spec)
        return encode_embed_result(
            summary=f"plot:{title or 'chart'}",
            artifacts=[{"kind": "plot", "title": title or "chart", "payload": payload}],
        )

    def embed_structure(
        title: str,
        format: str,
        content: str | None = None,
        path: str | None = None,
    ) -> str:
        """Embed a molvis structure viewer in the conversation UI."""
        from molexp.agent.ops.embed import encode_embed_result

        fmt = (format or "xyz").strip().lower().lstrip(".")
        if fmt not in ("xyz", "extxyz", "pdb"):
            return "error: format must be one of xyz|extxyz|pdb"
        body = (content or "").strip()
        if not body and path:
            from molexp.agent.ops.paths import safe_path

            # Honor chat confinement when LocalCodeEnv has _confine_rel.
            confine_rel = getattr(ctx.code, "_confine_rel", None)
            rel = confine_rel(path) if callable(confine_rel) else path
            try:
                candidate = safe_path(ctx.workspace_root, rel)
            except ValueError as exc:
                return f"error: {exc}"
            if not candidate.is_file():
                return f"error: path not found: {rel!r}"
            try:
                body = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                return f"error: cannot read {rel!r}: {exc}"
        if not body:
            return "error: provide content= or path= with structure text"
        # Cap size for wire events (chat inline).
        if len(body) > 512_000:
            return "error: structure content exceeds 512 KiB; write a shorter frame"
        fname = f"structure.{fmt}"
        return encode_embed_result(
            summary=f"structure:{title or fname}",
            artifacts=[
                {
                    "kind": "structure",
                    "title": title or fname,
                    "payload": {
                        "format": fmt,
                        "filename": fname,
                        "content": body,
                    },
                }
            ],
        )

    def run_land(
        project: str,
        experiment: str,
        run_id: str,
        files: str | None = None,
        sources: str | None = None,
        results_json: str | None = None,
    ) -> str:
        """Land workspace products onto a Run (full surface only)."""
        import json

        from molexp.agent.ops.land import land_run_outputs

        def _parse_list(raw: str | None) -> list[str]:
            if raw is None or not str(raw).strip():
                return []
            text = str(raw).strip()
            if text.startswith("["):
                data = json.loads(text)
                if not isinstance(data, list):
                    raise ValueError("list fields must be a JSON array of strings")
                return [str(x).strip() for x in data if str(x).strip()]
            return [p.strip() for p in text.split(",") if p.strip()]

        try:
            results: dict[str, object] | None = None
            if results_json and results_json.strip():
                parsed = json.loads(results_json)
                if not isinstance(parsed, dict):
                    return "error: results_json must be a JSON object"
                results = dict(parsed)
            summary = land_run_outputs(
                ctx.workspace_root,
                project=project,
                experiment=experiment,
                run_id=run_id,
                files=_parse_list(files),
                sources=_parse_list(sources),
                results=results,
            )
            return (
                f"ok run_id={summary['run_id']} status={summary['status']}\n"
                f"artifacts={summary['artifacts']}\n"
                f"sources={summary['sources']}\n"
                f"results={summary['results']}\n"
                f"path={summary['run_path']}"
            )
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def discover(query: str, kind: str | None = None) -> str:
        """Search builtins, knowledge, and/or the live MCP catalog."""
        hits = ctx.discovery.search(query, kind=kind)
        if not hits:
            if kind in (None, "builtin", "tool"):
                lines = ["(builtin tools always available:)"]
                lines.extend(
                    f"- {t.name}: {t.description[:120]}"
                    for t in builtin_tool_specs(surface=surface)
                )
                return "\n".join(lines)
            catalog = ctx.discovery.tools()
            if catalog and (kind in (None, "mcp_tool", "tool")):
                lines = ["(no knowledge hits; available MCP tools this session:)"]
                lines.extend(f"- {t.name}: {t.description[:120]}" for t in catalog[:40])
                return "\n".join(lines)
            return f"no matches for {query!r}"
        return "\n".join(f"{h.kind}\t{h.ref}\t{h.title}\t{h.summary}" for h in hits)

    def describe(ref: str) -> str:
        """Describe a discovery ref (builtin tool, MCP tool, or knowledge path)."""
        for spec in builtin_tool_specs(surface=surface):
            if spec.name == ref:
                return f"# tool {spec.name}\nsource: {spec.source}\n\n{spec.description}"
        return ctx.discovery.describe(ref)

    by_name: dict[str, Callable[..., str]] = {
        "workspace_ensure": workspace_ensure,
        "workspace_inspect": workspace_inspect,
        "code_write": code_write,
        "code_run": code_run,
        "embed_plot": embed_plot,
        "embed_structure": embed_structure,
        "run_land": run_land,
        "discover": discover,
        "describe": describe,
    }
    # Stable catalog order from the BUILTIN_TOOLS sequence. (The dropped
    # asserts here duplicated tests/test_agent/ops/test_ops_surface.py, which
    # already pins the built surface against BUILTIN_TOOL_NAMES per surface.)
    from molexp.agent.ops.builtins import BUILTIN_TOOLS

    return tuple(by_name[t.name] for t in BUILTIN_TOOLS if t.name in allowed)


def render_discovery_catalog(
    ctx: AgentSessionContext,
    *,
    surface: str = "chat",
) -> str:
    """System appendix: mounted builtins + live MCP catalog (if any)."""
    lines: list[str] = [
        f"## Builtin tools (surface={surface})",
    ]
    for s in builtin_tool_specs(surface=surface):
        desc = s.description.replace("\n", " ")[:120]
        lines.append(f"- `{s.name}`: {desc}")

    mcp: tuple[ToolSpec, ...] = ctx.discovery.tools()
    if mcp:
        lines.append("")
        lines.append("## MCP tools available this session (runtime catalog)")
        for s in mcp[:60]:
            desc = s.description.replace("\n", " ")[:100]
            lines.append(f"- `{s.name}`: {desc}")
        if len(mcp) > 60:
            lines.append(f"… ({len(mcp) - 60} more; use discover)")
    return "\n".join(lines)
