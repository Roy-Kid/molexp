"""Runtime catalog discovery — MCP tool list + workspace knowledge.

No hardcoded third-party API tables. MCP tool names come only from the
*live* toolset objects passed in at session/turn open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molexp.agent.ops.protocols import Hit, ToolSpec


class CatalogDiscovery:
    """Discovery over (1) open MCP toolsets and (2) workspace knowledge."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        mcp_toolsets: tuple[Any, ...] = (),
        mcp_tool_specs: tuple[ToolSpec, ...] = (),
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._toolsets = mcp_toolsets
        # Prefer explicit specs (from async list); else probe toolsets sync.
        self._specs = mcp_tool_specs

    def tools(self) -> tuple[ToolSpec, ...]:
        if self._specs:
            return self._specs
        # Best-effort: many toolsets expose .tools or similar only when entered.
        # Without async list at construction, return empty — still lawful:
        # names are never hardcoded; empty means "catalog not yet listed".
        return ()

    def search(self, query: str, *, kind: str | None = None) -> tuple[Hit, ...]:
        q = query.strip().lower()
        hits: list[Hit] = []
        if kind in (None, "mcp_tool", "tool"):
            for spec in self.tools():
                blob = f"{spec.name} {spec.description}".lower()
                if not q or q in blob:
                    hits.append(
                        Hit(
                            ref=spec.name,
                            kind="mcp_tool",
                            title=spec.name,
                            summary=spec.description[:200],
                            score=1.0 if q and q in spec.name.lower() else 0.5,
                        )
                    )
        if kind in (None, "knowledge"):
            hits.extend(self._search_knowledge(query))
        hits.sort(key=lambda h: h.score, reverse=True)
        return tuple(hits[:40])

    def describe(self, ref: str) -> str:
        for spec in self.tools():
            if spec.name == ref:
                return f"# tool {spec.name}\nsource: {spec.source}\n\n{spec.description}"
        # knowledge path
        try:
            from molexp.workspace import Bundle

            concept = Bundle(self._root).get(ref)
        except Exception as exc:
            return f"error: no detail for {ref!r} ({type(exc).__name__}: {exc})"
        try:
            meta = concept.read_meta()
            body = (concept.read_index() or "")[:8000]
            return f"# {ref}\ntype: {meta.get('type', '?')}\n\n{body}"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def _search_knowledge(self, query: str) -> list[Hit]:
        try:
            from molexp.workspace import Bundle
        except Exception:
            return []
        try:
            result = Bundle(self._root).search(query, limit=20)
        except Exception:
            return []
        rows: list[Hit] = []
        for hit in result.hits:
            entry = hit.entry
            rows.append(
                Hit(
                    ref=entry.path,
                    kind="knowledge",
                    title=entry.title or entry.path,
                    summary=(hit.snippet or "")[:200],
                    score=0.6,
                )
            )
        return rows
