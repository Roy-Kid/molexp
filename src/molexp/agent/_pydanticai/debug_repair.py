"""Source-grounded debug-loop repair — a single MCP-attached Agent.

When AuthorMode's per-task ``RunTaskDebugLoop`` stage finds a generated
test failing, the repair LLM has to figure out *why* — typically an
``AttributeError`` / ``ImportError`` / ``NameError`` showing the prior
codegen assumed a non-existent project API. Without source access the
repair is the same blind guess that produced the bug. This module wires
the repair through a ``pydantic_ai.Agent`` with the molmcp MCP toolset
attached. The prompt does not hardcode any specific tool name; it tells
the LLM it has a code-discovery MCP attached and describes the abstract
tool roles (catalog/browse vs lookup/introspect) so the LLM picks the
right tool from its own tool list — pydantic-ai already exposes the MCP
server's tool schemas to the model.

Public surface is a single closure factory — :func:`build_repair_callable`
— that returns an ``async def repair(prompt: str) -> GeneratedModule``.
The closure manages the MCP ``async with`` lifecycle and the per-call
``UsageLimits`` budget; from the caller's side it is just an opaque
``Callable[[str], Awaitable[GeneratedModule]]`` (stdlib type) so
``modes/author/`` can consume it without ever importing ``pydantic_ai``.

No new Protocol / wrapper class / factory module — the closure *is* the
abstraction. pydantic-ai's ``Agent`` covers tool dispatch, MCP
connection management, retries, and structured output natively; this
file is the wiring, not a reimplementation.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import cast

from mollog import get_logger
from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.usage import UsageLimits

from molexp.agent.modes.author.codegen import GeneratedModule

__all__ = ["build_repair_callable"]

_LOG = get_logger(__name__)

# pydantic-ai's ``Agent(model=...)`` accepts any of these shapes.
type PydanticAiModel = models.Model | models.KnownModelName | str

_DEFAULT_RETRIES = 3
"""``output_retries`` for the repair agent — covers transient MCP restarts
plus one structured-output validation slip."""

_DEFAULT_REQUEST_LIMIT = 30
"""Max model requests per repair call (one per MCP tool-call round-trip).
A grounded repair typically resolves in a handful of look-ups."""


_REPAIR_SYSTEM_PROMPT = (
    "You are a test-failure repair agent for the molexp / molcrafts "
    "project. You receive a failed task: the current implementation "
    "source, the test source, and the pytest traceback. Rewrite ONLY "
    "the implementation module so the test passes; return the full "
    "corrected module source.\n"
    "\n"
    "TOOLS — you have a set of tools attached from a code-discovery "
    "MCP server: the molcrafts toolchain catalog. Inspect your tool "
    "list before you start. The tools fall into roughly three roles "
    "— match by name pattern, not by exact identifier:\n"
    "  • CATALOG / OUTLINE — return a hierarchical map of a source's "
    "packages → modules → symbols, each carrying a one-line summary. "
    "Name usually contains 'outline', 'index', 'list', or 'tree'.\n"
    "  • CAPABILITY / SEARCH — take a natural-language task "
    "description, return ranked real-source matches (qualname, kind, "
    "signature, summary, examples). Name usually contains 'find', "
    "'search', or 'capability'.\n"
    "  • DETAIL / LOOKUP — take a fully-qualified name, return "
    "signature / docstring / source / relationships. Name usually "
    "contains 'describe', 'get', or 'inspect'.\n"
    "\n"
    "DISCOVERY PROTOCOL — when the traceback shows AttributeError, "
    "ImportError, ModuleNotFoundError, or NameError, the prior "
    "implementation assumed a non-existent project API. Find the "
    "right one by browsing the catalog, never by guessing:\n"
    "  Step 1 (RECOMMENDED FIRST CALL when the missing module is "
    "unfamiliar) — read the catalog. Call a CATALOG/OUTLINE tool for "
    "the relevant source (most often pkg:molpy) and skim each "
    "module's `summary` to locate the right place to look. Skip this "
    "step only when the failed import names a module you have already "
    "mapped in a prior tool call.\n"
    "  Step 2 — from the traceback, identify what capability the "
    "failed line was trying to invoke (e.g. 'assign OPLS-AA atom "
    "types', 'write a LAMMPS data file', 'build a polymer chain'). "
    "Use a CAPABILITY/SEARCH tool to discover ranked real-source "
    "matches.\n"
    "  Step 3 — pick the best match. Any returned hit (class / "
    "function / method) whose summary semantically matches IS usable "
    "— use the returned qualname verbatim as the import / attribute "
    "path in your patch. Don't demand a name-perfect match.\n"
    "  Step 4 — use a DETAIL/LOOKUP tool to confirm the chosen "
    "match's signature and docstring before patching. If the first "
    "search returns nothing usable, retry with different phrasings "
    "(verb synonyms, the underlying domain action, the file format).\n"
    "  Step 5 — patch the implementation to use the discovered API.\n"
    "\n"
    "Keep the module a molexp.workflow.Task subclass. Work "
    "economically — a typical fix takes a handful of tool calls. Do "
    "not write code outside the corrected implementation module."
)


@contextlib.contextmanager
def _silence_process_stdio() -> Iterator[None]:
    """Temporarily point stdout/stderr fds at ``os.devnull``.

    Some MCP servers print startup banners directly from the child
    process; redirecting Python's ``sys.stderr`` is not enough because
    the child inherits OS-level file descriptors.
    """
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def _build_repair_agent(
    model: PydanticAiModel,
    *,
    toolsets: tuple[MCPToolset, ...] = (),
    tools: tuple[Callable[..., object], ...] = (),
    output_retries: int = _DEFAULT_RETRIES,
) -> Agent[None, GeneratedModule]:
    """Construct the source-grounded repair agent.

    ``toolsets`` carries the molmcp ``MCPToolset`` in production;
    ``tools`` lets tests inject plain fake source-introspection
    callables instead of spawning a real MCP subprocess.
    """
    agent = Agent(
        model=model,
        output_type=GeneratedModule,
        system_prompt=_REPAIR_SYSTEM_PROMPT,
        toolsets=list(toolsets),
        tools=list(tools),
        output_retries=output_retries,
    )
    return cast("Agent[None, GeneratedModule]", agent)


def build_repair_callable(
    *,
    workspace: Path | None,
    model: object,
    retries: int = _DEFAULT_RETRIES,
    request_limit: int = _DEFAULT_REQUEST_LIMIT,
) -> Callable[[str], Awaitable[GeneratedModule]] | None:
    """Build a source-grounded repair callable, or ``None`` when molmcp is absent.

    Reads the molmcp ``stdio`` MCP-server entry from the workspace's
    :class:`~molexp.agent.mcp.store.McpStore` (the user-scope store is
    still consulted when ``workspace`` is ``None``), constructs an MCP-
    attached ``pydantic_ai.Agent`` whose output type is :class:`GeneratedModule`,
    and returns a closure that drives it once per call.

    Args:
        workspace: Workspace root used to resolve the MCP store; user-
            scope is consulted when ``None``.
        model: pydantic-ai model the repair agent runs on — passed as
            ``object`` so callers outside ``_pydanticai/`` (AuthorMode)
            need not import the pydantic-ai model alias.
        retries: ``output_retries`` for the repair agent.
        request_limit: Max model requests for one repair call.

    Returns:
        An ``async def repair(prompt: str) -> GeneratedModule`` callable,
        or ``None`` when no molmcp ``stdio`` entry is configured (the
        caller then falls back to the existing no-tool router path).
    """
    try:
        from molexp.agent.mcp.store import McpStore

        store = McpStore(workspace if workspace is not None else Path())
        entries = store.list()
    except OSError as exc:  # pragma: no cover — read-only fs / schema drift
        _LOG.warning(f"[debug-repair] could not read MCP store: {exc!r}")
        return None

    for entry in entries:
        if (
            entry.name == "molmcp"
            and entry.transport == "stdio"
            and entry.valid
            and not entry.shadowed
            and entry.command
        ):
            _LOG.debug(f"[debug-repair] wiring molmcp repair via {entry.command!r}")
            server = MCPToolset(
                StdioTransport(
                    command=entry.command,
                    args=list(entry.args),
                    env=None,
                )
            )
            agent = _build_repair_agent(
                cast("PydanticAiModel", model),
                toolsets=(server,),
                output_retries=retries,
            )
            return _wrap_agent_as_callable(agent, request_limit=request_limit)
    _LOG.debug("[debug-repair] no molmcp stdio server configured")
    return None


def _wrap_agent_as_callable(
    agent: Agent[None, GeneratedModule], *, request_limit: int
) -> Callable[[str], Awaitable[GeneratedModule]]:
    """Wrap an MCP-attached repair agent as an opaque async callable.

    The wrapper manages the per-call ``async with agent`` lifecycle and
    the ``UsageLimits`` budget. Failures surface as exceptions — the
    debug loop is expected to record a repair iteration and retry on
    the next attempt rather than silently swallow the error.
    """

    async def repair(prompt: str) -> GeneratedModule:
        with _silence_process_stdio():
            async with agent:
                result = await agent.run(
                    prompt,
                    usage_limits=UsageLimits(request_limit=request_limit),
                )
        return result.output

    return repair
