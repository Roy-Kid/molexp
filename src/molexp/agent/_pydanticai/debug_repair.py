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

The repair agent returns a :class:`~molexp.agent.modes.author.codegen.RepairDecision`
— diagnosis + an optional repaired impl + an optional repaired test
module. The agent decides per-failure whether the impl, the test, or
both need patching; the caller routes each through the same gates the
initial codegen uses.

Public surface is a single closure factory — :func:`build_repair_callable`
— that returns an ``async def repair(prompt: str) -> RepairDecision``.
The closure manages the MCP ``async with`` lifecycle, the per-call
``UsageLimits`` budget, and a wall-clock timeout so a stuck MCP tool
loop fails the repair instead of hanging the run; from the caller's
side it is just an opaque ``Callable[[str], Awaitable[RepairDecision]]``
(stdlib type) so ``modes/author/`` can consume it without ever
importing ``pydantic_ai``.

No new Protocol / wrapper class / factory module — the closure *is* the
abstraction. pydantic-ai's ``Agent`` covers tool dispatch, MCP
connection management, retries, and structured output natively; this
file is the wiring, not a reimplementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import cast

from mollog import get_logger
from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.usage import UsageLimits

from molexp.agent.modes.author.codegen import RepairDecision

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

_DEFAULT_TIMEOUT_SECONDS = 180.0
"""Wall-clock budget for one full ``agent.run`` (across all tool calls
+ output retries). Without a hard deadline a repair agent stuck in an
MCP tool-call loop can hang the run indefinitely; this turns that into
a deterministic failed-iteration."""


_REPAIR_SYSTEM_PROMPT = (
    "You are a test-failure repair agent. You receive a failed task: "
    "the current implementation source, the test source, the pytest "
    "traceback, and the PlanStep that drives the codegen layer. Either "
    "the impl OR the test (or both) may carry the bug — your job is "
    "to diagnose the root cause and emit a `RepairDecision` targeting "
    "whichever file needs to change.\n"
    "\n"
    "DIAGNOSIS FIRST. Always populate `diagnosis` with one sentence "
    "naming the root-cause file and the underlying mistake. Then "
    "decide which patch to emit:\n"
    "  - error happens BEFORE the task function is called (broken "
    "fixture, stub-class instantiation error, import error in the "
    "test module) → TEST bug → emit `test_source`\n"
    "  - the impl raises an exception or returns the wrong shape, "
    "and the test's assertion is reasonable given the PlanStep's "
    "declared inputs/outputs → IMPL bug → emit `impl`\n"
    "  - the test over-specifies content the PlanStep doesn't actually "
    "promise (exact whitespace, exact numeric value, specific item "
    "ordering) → TEST bug → emit `test_source` with shape-level "
    "assertions instead\n"
    "  - both files need work → emit both\n"
    "\n"
    "IMPL PATCH SHAPE — when you emit `impl`, the codegen layer "
    "assembles a function around your draft. The assembled module is:\n"
    "  ```\n"
    "  \"\"\"<PlanStep.composition_notes>\"\"\"\n"
    "  <imports>\n"
    "\n"
    "  async def <task_function>(ctx):\n"
    "      <input_name1> = ctx.inputs[...]   # auto-bound\n"
    "      <input_name2> = ctx.inputs[...]   # auto-bound\n"
    "      <YOUR BODY>\n"
    "      return {\"<output_name>\": <output_name>, ...}   # auto\n"
    "  ```\n"
    "RepairDecision fields:\n"
    "  - `imports`: tuple of import lines used in the body; each "
    "entry is one `import X` or `from X import Y`. Only symbols "
    "verified against the project source via your discovery tools "
    "(see protocol below).\n"
    "  - `body`: Python statements that compose the imports to bind "
    "the local names declared in PlanStep.io.outputs. No leading "
    "indentation; the assembler indents.\n"
    "\n"
    "TEST PATCH SHAPE — when you emit `test_source`, write the full "
    "pytest module source. Keep assertions SHAPE-LEVEL (dict keys, "
    "value types, non-emptiness, simple counts); avoid asserting "
    "exact substrings / numerics / ordering that the impl might pick "
    "differently. Imports use stdlib + pytest + symbols from the "
    "ALLOWED PROJECT IMPORTS list.\n"
    "\n"
    "TOOLS — you have a set of tools attached from a code-discovery MCP "
    "server exposing one or more project sources. Inspect your tool list "
    "before you start; the catalog itself tells you which packages "
    "exist. The tools fall into roughly three roles — match by name "
    "pattern, not by exact identifier:\n"
    "  - CATALOG / OUTLINE — return a hierarchical map of a source's "
    "packages -> modules -> symbols, each carrying a one-line summary. "
    "Name usually contains 'outline', 'index', 'list', or 'tree'.\n"
    "  - CAPABILITY / SEARCH — take a natural-language task "
    "description, return ranked real-source matches (qualname, kind, "
    "signature, summary, examples). Name usually contains 'find', "
    "'search', or 'capability'.\n"
    "  - DETAIL / LOOKUP — take a fully-qualified name, return "
    "signature / docstring / source / relationships. Name usually "
    "contains 'describe', 'get', or 'inspect'.\n"
    "\n"
    "DISCOVERY PROTOCOL — when the traceback shows AttributeError, "
    "ImportError, ModuleNotFoundError, or NameError, the prior "
    "implementation assumed a non-existent project API. Find the right "
    "one by browsing the catalog, never by guessing:\n"
    "  Step 1 (RECOMMENDED FIRST CALL when the missing module is "
    "unfamiliar) — read the catalog. Call a CATALOG/OUTLINE tool for "
    "the source whose top-level name appears in the failed import, and "
    "skim each module's `summary` to locate the right place to look. "
    "Skip this step only when the failed import names a module you have "
    "already mapped in a prior tool call.\n"
    "  Step 2 — from the traceback, identify what capability the failed "
    "line was trying to invoke (e.g. 'assign OPLS-AA atom types', "
    "'write a LAMMPS data file', 'build a polymer chain'). Use a "
    "CAPABILITY/SEARCH tool to discover ranked real-source matches.\n"
    "  Step 3 — pick the best match. Any returned hit (class / function "
    "/ method) whose summary semantically matches IS usable — use the "
    "returned qualname verbatim in `imports`. Don't demand a "
    "name-perfect match.\n"
    "  Step 4 — use a DETAIL/LOOKUP tool to confirm the chosen match's "
    "signature and docstring before patching. If the first search "
    "returns nothing usable, retry with different phrasings (verb "
    "synonyms, the underlying domain action, the file format).\n"
    "  Step 5 — write the body using the discovered API.\n"
    "\n"
    "Work economically — a typical fix takes a handful of tool calls."
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
) -> Agent[None, RepairDecision]:
    """Construct the source-grounded repair agent.

    ``toolsets`` carries the molmcp ``MCPToolset`` in production;
    ``tools`` lets tests inject plain fake source-introspection
    callables instead of spawning a real MCP subprocess.
    """
    agent = Agent(
        model=model,
        output_type=RepairDecision,
        system_prompt=_REPAIR_SYSTEM_PROMPT,
        toolsets=list(toolsets),
        tools=list(tools),
        output_retries=output_retries,
    )
    return cast("Agent[None, RepairDecision]", agent)


def build_repair_callable(
    *,
    workspace: Path | None,
    model: object,
    retries: int = _DEFAULT_RETRIES,
    request_limit: int = _DEFAULT_REQUEST_LIMIT,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Callable[[str], Awaitable[RepairDecision]] | None:
    """Build a source-grounded repair callable, or ``None`` when molmcp is absent.

    Reads the molmcp ``stdio`` MCP-server entry from the workspace's
    :class:`~molexp.agent.mcp.store.McpStore` (the user-scope store is
    still consulted when ``workspace`` is ``None``), constructs an MCP-
    attached ``pydantic_ai.Agent`` whose output type is :class:`RepairDecision`,
    and returns a closure that drives it once per call.

    Args:
        workspace: Workspace root used to resolve the MCP store; user-
            scope is consulted when ``None``.
        model: pydantic-ai model the repair agent runs on — passed as
            ``object`` so callers outside ``_pydanticai/`` (AuthorMode)
            need not import the pydantic-ai model alias.
        retries: ``output_retries`` for the repair agent.
        request_limit: Max model requests for one repair call.
        timeout_seconds: Wall-clock budget for one repair call. The
            ``asyncio.wait_for`` wrapper cancels the call on overrun
            and surfaces ``asyncio.TimeoutError`` so the caller treats
            it as a failed iteration.

    Returns:
        An ``async def repair(prompt: str) -> RepairDecision`` callable,
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
            return _wrap_agent_as_callable(
                agent,
                request_limit=request_limit,
                timeout_seconds=timeout_seconds,
            )
    _LOG.debug("[debug-repair] no molmcp stdio server configured")
    return None


def _wrap_agent_as_callable(
    agent: Agent[None, RepairDecision],
    *,
    request_limit: int,
    timeout_seconds: float,
) -> Callable[[str], Awaitable[RepairDecision]]:
    """Wrap an MCP-attached repair agent as an opaque async callable.

    The wrapper manages the per-call ``async with agent`` lifecycle, the
    ``UsageLimits`` budget, and a wall-clock deadline via
    ``asyncio.wait_for``. Failures (including timeouts) surface as
    exceptions — the debug loop is expected to record a repair iteration
    and retry on the next attempt rather than silently swallow the error.

    Concurrent invocations are serialised by an ``asyncio.Lock``:
    ``_silence_process_stdio`` swaps process-global fds, and
    ``stages.py`` fans out repair calls via ``asyncio.gather``. Without
    the lock two overlapping invocations capture each other's redirected
    fds and permanently silence the parent process. Serialising the
    repair callable also matches pydantic-ai's ``Agent`` lifecycle: the
    MCP refcount expects a single owner per ``async with`` scope.
    """
    repair_lock = asyncio.Lock()

    async def repair(prompt: str) -> RepairDecision:
        async def _run() -> RepairDecision:
            with _silence_process_stdio():
                async with agent:
                    result = await agent.run(
                        prompt,
                        usage_limits=UsageLimits(request_limit=request_limit),
                    )
            return result.output

        async with repair_lock:
            return await asyncio.wait_for(_run(), timeout=timeout_seconds)

    return repair
