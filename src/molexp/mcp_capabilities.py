"""Prefetch a harness ``CapabilityRegistry`` from the molmcp MCP server.

An application-tier bridge (shared by the ``molexp plan`` CLI and the server's
plan-task route) between the externally-provisioned **molmcp** MCP server and the
harness capability-grounding contract. It opens a stdio MCP session to the
configured ``molmcp`` server, enumerates the relevant molcrafts packages' symbols
via molmcp's ``find_capability`` discovery tool, and maps each returned node into
a harness :class:`~molexp.harness.schemas.ToolCapability`. The resulting
:class:`~molexp.harness.registry.in_memory.InMemoryCapabilityRegistry` snapshot is
handed to ``Mode.run(capability_registry=…)`` so the **harness stays MCP-free**
(it imports only ``agent.router``); every line of MCP I/O lives here.

It lives at the package root (alongside ``git`` / ``ids`` / ``atomicio``) rather
than under ``cli`` or ``server``: both application shells import it, and a sibling
``server → cli`` edge would be a layer smell. Every business-layer import is
deferred into a function body, so module load pulls only ``mollog`` + stdlib —
``import molexp`` stays light.

molmcp is *externally provisioned* — never a molexp dependency. When it is not
configured, the resolvers return ``None`` (grounding off) after a visible notice,
and the harness skips capability-aware validation. That downgrade is loud and
explicit — never a silent fallback. Two resolvers share the mapping + notice
logic: :func:`resolve_capability_registry` (sync, for the CLI) and
:func:`aresolve_capability_registry` (async, for the server's async route).

The catalog is built by opening molmcp, **listing tools at runtime**, ranking
discovery-shaped tools by name substring, and querying them with the experiment
``task`` draft (or an explicit ``queries`` list). No compiled-in polymer/LAMMPS
query menu — auto-discovery law. Phase 1b feeds the same catalog to the
``bound_workflow_binder`` agent so binder and validator stay self-consistent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mollog import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import ToolCapability

_LOG = get_logger(__name__)

DEFAULT_SERVER_NAME = "molmcp"

#: Legacy alias kept for call-site imports. **Empty by design** — plan grounding
#: no longer ships a domain-specific polymer/LAMMPS query table. Pass
#: ``task=`` (the experiment draft) or an explicit ``queries=`` sequence;
#: discovery tool **names** are chosen from the live MCP ``list_tools``
#: catalog (auto-discovery law).
DEFAULT_CAPABILITY_QUERIES: tuple[str, ...] = ()

_SKIP_PARAMS = frozenset({"self", "cls"})


# ── Signature parsing (pure) ───────────────────────────────────────────────


def _split_top_level(params: str) -> list[str]:
    """Split a parameter list on top-level commas (bracket-depth aware).

    Commas inside ``dict[str, Any]`` / ``= [1, 2]`` defaults stay with their
    parameter rather than splitting it.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_signature_params(
    signature: str | None,
) -> tuple[list[str], list[str], bool] | None:
    """Parse a rendered signature into ``(names, required, accepts_extra_keys)``.

    ``None`` when the signature is absent or unparseable (e.g. a class node whose
    ``signature`` is ``null``) — the caller then emits a permissive schema.
    ``required`` excludes parameters that carry a default. The ``self`` / ``cls``
    receiver and ``*args`` are dropped; positional-only (``/``) and keyword-only
    (``*``) markers are ignored. ``accepts_extra_keys`` is True when the signature
    has ``**kwargs`` — the capability then takes arbitrary keyword keys, so the
    schema must not restrict them.
    """
    if not signature:
        return None
    open_idx = signature.find("(")
    close_idx = signature.rfind(")")
    if open_idx == -1 or close_idx <= open_idx:
        return None
    names: list[str] = []
    required: list[str] = []
    accepts_extra = False
    for raw in _split_top_level(signature[open_idx + 1 : close_idx]):
        if raw in ("/", "*"):
            continue
        if raw.startswith("**"):
            accepts_extra = True  # **kwargs — arbitrary keyword keys allowed
            continue
        if raw.startswith("*"):
            continue  # *args — positional varargs, not a named keyword
        has_default = "=" in raw
        name = raw.split(":", 1)[0].split("=", 1)[0].strip()
        if not name or name in _SKIP_PARAMS:
            continue
        names.append(name)
        if not has_default:
            required.append(name)
    return names, required, accepts_extra


def synthesize_input_schema(signature: str | None) -> dict[str, object]:
    """Build a shallow ``input_schema`` from a rendered signature string.

    The harness validator only key-checks (provided keys ⊆ ``properties``;
    ``required`` ⊆ provided), so a name-level schema suffices. A schema with NO
    ``properties`` key is the validator's wildcard ("any input allowed") — which
    is what unparseable / class-constructor signatures and ``**kwargs`` functions
    get, so they never false-reject a bound call. ``required`` is enforced
    regardless.
    """
    parsed = parse_signature_params(signature)
    if parsed is None:
        return {"type": "object"}
    names, required, accepts_extra = parsed
    schema: dict[str, object] = {"type": "object", "required": required}
    if not accepts_extra:
        schema["properties"] = {name: {} for name in names}
    return schema


# ── Node → ToolCapability mapping (pure) ───────────────────────────────────


def capability_from_node(
    node: Mapping[str, object],
    *,
    snapshot_commit: str | None = None,
) -> ToolCapability | None:
    """Map one molmcp discovery ``node`` to a harness :class:`ToolCapability`.

    ``None`` when the node carries no ``qualname`` (nothing to bind to). The
    dotted ``qualname`` is both the capability ``id`` and its ``callable_path``;
    ``package`` is its first segment; ``input_schema`` is synthesized from the
    node's rendered ``signature``; ``version`` records molmcp's snapshot commit
    for provenance.
    """
    from molexp.harness.schemas import ToolCapability

    qualname = node.get("qualname")
    if not isinstance(qualname, str) or not qualname:
        return None
    name = node.get("name")
    summary = node.get("summary")
    kind = node.get("kind")
    signature = node.get("signature")
    return ToolCapability(
        id=qualname,
        package=qualname.split(".", 1)[0],
        name=name if isinstance(name, str) and name else qualname.rsplit(".", 1)[-1],
        description=summary if isinstance(summary, str) else "",
        input_schema=synthesize_input_schema(signature if isinstance(signature, str) else None),
        output_schema={},
        callable_path=qualname,
        supported_backends=["local"],
        tags=[kind] if isinstance(kind, str) and kind else [],
        version=snapshot_commit,
    )


def _iter_nodes(
    payload: Mapping[str, object],
) -> Iterable[tuple[Mapping[str, object], str | None]]:
    """Yield ``(node, snapshot_commit)`` pairs from a molmcp tool-result payload.

    Tolerates both molmcp result shapes: ``find_capability`` (``"matches"`` whose
    items wrap a ``"node"``) and ``search_symbols`` / ``outline`` (a flat
    ``"results"`` / ``"symbols"`` list whose items are node dicts).
    """
    from collections.abc import Mapping as _Mapping
    from typing import cast

    def as_map(value: object) -> Mapping[str, object] | None:
        return cast("Mapping[str, object]", value) if isinstance(value, _Mapping) else None

    commit: str | None = None
    snapshot = as_map(payload.get("snapshot"))
    if snapshot is not None:
        commit_raw = snapshot.get("commit")
        commit = commit_raw if isinstance(commit_raw, str) else None

    matches = payload.get("matches")
    if isinstance(matches, list):
        for match in matches:
            match_map = as_map(match)
            if match_map is not None:
                node = as_map(match_map.get("node"))
                if node is not None:
                    yield node, commit
        return
    for key in ("results", "symbols", "nodes"):
        items = payload.get(key)
        if isinstance(items, list):
            for item in items:
                item_map = as_map(item)
                if item_map is not None:
                    inner = as_map(item_map.get("node"))
                    yield (inner if inner is not None else item_map), commit
            return


def capabilities_from_payloads(
    payloads: Iterable[Mapping[str, object]],
) -> list[ToolCapability]:
    """Map molmcp tool-result payloads to a capability list, deduped by id."""
    by_id: dict[str, ToolCapability] = {}
    for payload in payloads:
        for node, commit in _iter_nodes(payload):
            cap = capability_from_node(node, snapshot_commit=commit)
            if cap is not None and cap.id not in by_id:
                by_id[cap.id] = cap
    return list(by_id.values())


# ── molmcp MCP session (async I/O) ─────────────────────────────────────────


def _payload_from_result(result: object) -> Mapping[str, object] | None:
    """Extract the JSON dict from an MCP ``CallToolResult``'s text content."""
    if not hasattr(result, "content"):
        return None
    content = result.content
    if not content:
        return None
    for block in content:
        text = block.text if hasattr(block, "text") else None
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _rank_discovery_tools(tool_names: Sequence[str]) -> list[str]:
    """Order live MCP tool names by discovery usefulness (no fixed package list).

    Matching is by **substring** on whatever the server advertises today —
    ``find_capability``, ``explore``, ``search``, ``outline``, … — so molmcp
    renames still work without a molexp source edit when the role is clear.
    """
    scored: list[tuple[int, str]] = []
    for name in tool_names:
        lower = name.lower()
        if "find_capability" in lower:
            scored.append((0, name))
        elif "explore" in lower:
            scored.append((1, name))
        elif "search_symbols" in lower or lower.endswith("_search") or "search" in lower:
            scored.append((2, name))
        elif "outline" in lower:
            scored.append((3, name))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in scored]


def _call_args_for_discovery_tool(
    tool_name: str, query: str, *, max_results: int
) -> dict[str, object]:
    """Build best-effort kwargs for a discovery tool (schema varies by server)."""
    lower = tool_name.lower()
    if "find_capability" in lower or "explore" in lower:
        return {"task": query, "budget_chars": 16000, "max_results": max_results}
    if "outline" in lower:
        return {"source": query} if query and not query.startswith("public") else {}
    return {"query": query, "limit": max_results}


async def fetch_molmcp_capabilities(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    task: str | None = None,
    queries: Sequence[str] | None = None,
    max_results: int = 12,
) -> list[ToolCapability]:
    """Open a stdio session to molmcp and prefetch a deduped capability catalog.

    **Auto-discovery law:** discovery tool names are taken from the live
    MCP ``list_tools`` catalog; query text comes from ``task`` (experiment
    draft) or an explicit ``queries`` sequence — never from a compiled-in
    polymer/LAMMPS table.

    Raises:
        LookupError: if ``server_name`` is not configured or is not a stdio
            server (the sync wrapper turns this into a notice + ``None``).
    """
    import os
    from pathlib import Path

    # ``mcp`` arrives transitively via the optional ``agent`` extra (pydantic-ai)
    # and molmcp is externally provisioned, so this stays a lazy, in-function
    # import; ty lacks stubs for the submodule.
    from mcp import ClientSession, StdioServerParameters  # ty: ignore[unresolved-import]
    from mcp.client.stdio import stdio_client  # ty: ignore[unresolved-import]

    from molexp.agent.mcp import McpScope, McpStore

    store = McpStore(workspace_root)
    entry = store.get(McpScope.WORKSPACE, server_name) or store.get(McpScope.USER, server_name)
    if entry is None:
        raise LookupError(f"MCP server {server_name!r} is not configured")
    spec = store.resolve(entry)
    if spec.transport != "stdio" or not spec.command:
        raise LookupError(f"MCP server {server_name!r} is not a stdio server")

    if queries:
        query_list = [q.strip() for q in queries if q and str(q).strip()]
    elif task and task.strip():
        query_list = [task.strip()]
    else:
        # Single domain-agnostic probe — not a hand-maintained science menu.
        query_list = ["public package APIs and executable capabilities"]

    params = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=dict(spec.env) or None,
    )
    payloads: list[Mapping[str, object]] = []
    # Route the server's stderr (FastMCP startup banner + logs) to /dev/null so
    # it never bleeds into the CLI's own output.
    with Path(os.devnull).open("w", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            tool_names = [t.name for t in listed.tools] if hasattr(listed, "tools") else []
            if not tool_names and isinstance(listed, list):
                tool_names = [
                    t.name for t in listed if hasattr(t, "name") and isinstance(t.name, str)
                ]
            candidates = _rank_discovery_tools(tool_names)
            if not candidates:
                _LOG.warning(
                    "[mcp_capabilities] no discovery-shaped tools in MCP catalog; "
                    f"tools={tool_names[:20]!r}"
                )
            for tool_name in candidates[:4]:
                for query in query_list:
                    args = _call_args_for_discovery_tool(tool_name, query, max_results=max_results)
                    try:
                        result = await session.call_tool(tool_name, args)
                    except Exception as exc:
                        _LOG.debug(f"[mcp_capabilities] {tool_name!r} query failed: {exc!r}")
                        continue
                    payload = _payload_from_result(result)
                    if payload is not None:
                        payloads.append(payload)
    return capabilities_from_payloads(payloads)


def _log_notice(message: str) -> None:
    _LOG.warning(message)


def _registry_from_caps(
    caps: list[ToolCapability], *, server_name: str, say: Callable[[str], None]
) -> CapabilityRegistry | None:
    """Build the snapshot registry from prefetched caps (loud on empty)."""
    from molexp.harness.registry import InMemoryCapabilityRegistry

    if not caps:
        say("capability grounding off — molmcp returned no capabilities")
        return None
    say(f"capability grounding on — {len(caps)} molcrafts capabilities via {server_name}")
    return InMemoryCapabilityRegistry(caps)


def _notice_for_prefetch_error(exc: Exception, say: Callable[[str], None]) -> None:
    """Emit the loud, explicit ground-off notice for a prefetch failure."""
    if isinstance(exc, LookupError):
        say(f"capability grounding off — {exc} (binding will not be validated against molpy)")
    else:
        say(f"capability grounding off — molmcp prefetch failed: {exc}")


def resolve_capability_registry(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    task: str | None = None,
    queries: Sequence[str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> CapabilityRegistry | None:
    """Build a grounded ``CapabilityRegistry`` from molmcp, or ``None`` (loud).

    Synchronous entry (for the CLI). Prefer ``task=`` (experiment draft) so
    grounding follows the user request instead of a fixed query menu. On any
    miss (molmcp unconfigured / unreachable / empty) emits a visible notice
    via ``notify`` and returns ``None`` — never a silent downgrade.
    """
    import asyncio

    say = notify if notify is not None else _log_notice
    try:
        caps = asyncio.run(
            fetch_molmcp_capabilities(
                workspace_root,
                server_name=server_name,
                task=task,
                queries=queries,
            )
        )
    except Exception as exc:  # prefetch is best-effort: report and proceed ungrounded
        _notice_for_prefetch_error(exc, say)
        return None
    return _registry_from_caps(caps, server_name=server_name, say=say)


async def aresolve_capability_registry(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    task: str | None = None,
    queries: Sequence[str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> CapabilityRegistry | None:
    """Async sibling of :func:`resolve_capability_registry` for an async caller.

    Awaits the prefetch directly (rather than ``asyncio.run``), so it is safe to
    call from inside a running event loop — e.g. the server's async plan-task
    route. Same loud-on-miss / never-silent contract.
    """
    say = notify if notify is not None else _log_notice
    try:
        caps = await fetch_molmcp_capabilities(
            workspace_root,
            server_name=server_name,
            task=task,
            queries=queries,
        )
    except Exception as exc:  # prefetch is best-effort: report and proceed ungrounded
        _notice_for_prefetch_error(exc, say)
        return None
    return _registry_from_caps(caps, server_name=server_name, say=say)


def _merge_curation_built_ins(science: CapabilityRegistry | None) -> CapabilityRegistry:
    """Seed a concrete registry with the grounded science caps + curation built-ins.

    ``register`` is concrete-only (not on the ``CapabilityRegistry`` Protocol), so
    the merge holds a fresh ``InMemoryCapabilityRegistry`` rather than mutating the
    Protocol-typed science return. Curation ids are ``molexp.curation.*`` and never
    collide with molmcp science ids.
    """
    from molexp.harness.capabilities import curation_capabilities, lifecycle_capabilities
    from molexp.harness.registry import InMemoryCapabilityRegistry

    merged = InMemoryCapabilityRegistry()
    if science is not None:
        for cap in science.list_capabilities():
            merged.register(cap)
    for cap in curation_capabilities():
        merged.register(cap)
    for cap in lifecycle_capabilities():
        merged.register(cap)
    return merged


def resolve_curation_capability_registry(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    task: str | None = None,
    queries: Sequence[str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> CapabilityRegistry:
    """Merge the built-in curation catalog onto the molmcp-grounded science registry.

    Synchronous entry (for the CLI). Grounds science via
    :func:`resolve_capability_registry` (loud, may be ``None`` when molmcp is off),
    then registers every ``curation_capabilities()`` + ``lifecycle_capabilities()`` built-in. **Unlike the science
    resolver this never returns ``None``** — the ``molexp.curation.*`` built-ins are
    always present even with science grounding off.
    """
    science = resolve_capability_registry(
        workspace_root,
        server_name=server_name,
        task=task,
        queries=queries,
        notify=notify,
    )
    return _merge_curation_built_ins(science)


async def aresolve_curation_capability_registry(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    task: str | None = None,
    queries: Sequence[str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> CapabilityRegistry:
    """Async sibling of :func:`resolve_curation_capability_registry`.

    Awaits :func:`aresolve_capability_registry` for the science half (safe inside a
    running event loop — e.g. the curate-tasks route), then merges the curation
    built-ins. Never returns ``None``.
    """
    science = await aresolve_capability_registry(
        workspace_root,
        server_name=server_name,
        task=task,
        queries=queries,
        notify=notify,
    )
    return _merge_curation_built_ins(science)
