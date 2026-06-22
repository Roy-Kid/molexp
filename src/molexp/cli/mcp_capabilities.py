"""Prefetch a harness ``CapabilityRegistry`` from the molmcp MCP server.

App-layer (CLI) bridge between the externally-provisioned **molmcp** MCP server
and the harness capability-grounding contract. It opens a stdio MCP session to
the configured ``molmcp`` server, enumerates the relevant molcrafts packages'
symbols via molmcp's ``find_capability`` discovery tool, and maps each returned
node into a harness :class:`~molexp.harness.schemas.ToolCapability`. The
resulting :class:`~molexp.harness.registry.in_memory.InMemoryCapabilityRegistry`
snapshot is handed to ``Mode.run(capability_registry=…)`` so the **harness stays
MCP-free** (it imports only ``agent.router``); every line of MCP I/O lives here,
at the application layer.

molmcp is *externally provisioned* — never a molexp dependency. When it is not
configured, :func:`resolve_capability_registry` returns ``None`` (grounding off)
after a visible notice, and the harness skips capability-aware validation. That
downgrade is loud and explicit — never a silent fallback.

The catalog is built from a fixed set of domain-spanning ``find_capability``
queries (the union, deduped by symbol id). Phase 1b feeds the *same* catalog to
the ``bound_workflow_binder`` agent, so the binder only ever names symbols that
are in the catalog — keeping the binder's choices and the validator's
existence/shape checks self-consistent by construction.
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

#: Domain-spanning capability queries. molmcp ranks symbols against each; the
#: deduped union becomes the grounded catalog.
DEFAULT_CAPABILITY_QUERIES: tuple[str, ...] = (
    "build a coarse-grained molecule from beads",
    "create a coarse-grained bead with a charge",
    "bond two coarse-grained beads together",
    "build a polymer chain from monomers",
    "replicate or tile a molecular structure",
    "convert a coarse-grained structure to a frame",
    "define a periodic simulation box",
    "pack molecules into a simulation box",
    "write a LAMMPS data file",
    "write a complete LAMMPS system with a force field",
    "export a structure to a GROMACS topology",
)

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
    content = getattr(result, "content", None)
    if not content:
        return None
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


async def fetch_molmcp_capabilities(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    queries: Sequence[str] = DEFAULT_CAPABILITY_QUERIES,
    max_results: int = 12,
) -> list[ToolCapability]:
    """Open a stdio session to molmcp and prefetch a deduped capability catalog.

    Resolves the ``server_name`` entry from molexp's MCP config store, spawns its
    stdio server, runs each query through ``molmcp_find_capability``, and maps the
    returned nodes to capabilities.

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
            for query in queries:
                result = await session.call_tool(
                    "molmcp_find_capability",
                    {"task": query, "max_results": max_results},
                )
                payload = _payload_from_result(result)
                if payload is not None:
                    payloads.append(payload)
    return capabilities_from_payloads(payloads)


def _log_notice(message: str) -> None:
    _LOG.warning(message)


def resolve_capability_registry(
    workspace_root: str | Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    queries: Sequence[str] = DEFAULT_CAPABILITY_QUERIES,
    notify: Callable[[str], None] | None = None,
) -> CapabilityRegistry | None:
    """Build a grounded ``CapabilityRegistry`` from molmcp, or ``None`` (loud).

    Synchronous entry for the CLI. Runs the async prefetch; on any miss (molmcp
    unconfigured / unreachable / empty) it emits a visible notice via ``notify``
    (default: a logger warning) and returns ``None`` so the caller proceeds
    ungrounded — an explicit, never-silent downgrade.
    """
    import asyncio

    from molexp.harness import InMemoryCapabilityRegistry

    say = notify if notify is not None else _log_notice
    try:
        caps = asyncio.run(
            fetch_molmcp_capabilities(workspace_root, server_name=server_name, queries=queries)
        )
    except LookupError as exc:
        say(f"capability grounding off — {exc} (binding will not be validated against molpy)")
        return None
    except Exception as exc:  # prefetch is best-effort: report and proceed ungrounded
        say(f"capability grounding off — molmcp prefetch failed: {exc}")
        return None
    if not caps:
        say("capability grounding off — molmcp returned no capabilities")
        return None
    say(f"capability grounding on — {len(caps)} molcrafts capabilities via {server_name}")
    return InMemoryCapabilityRegistry(caps)
