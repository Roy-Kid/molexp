"""``build_research_planner`` — the sole pydantic-ai agent for PlanMode's plan stage.

PlanMode's :class:`~molexp.agent.modes.plan.stages.research_and_plan.ResearchAndPlan`
stage delegates every model + tool interaction to a single
``pydantic_ai.Agent`` built here. The agent is given an MCP toolset for
the molcrafts source catalog; pydantic-ai drives the model ↔ tool loop
end-to-end (no surrounding Python iteration). The agent's structured
output type is :class:`~molexp.agent.modes._planning.PlanGraph` —
``api_refs`` + ``composition_notes`` live inline on each
:class:`~molexp.agent.modes._planning.PlanStep`, so there is no separate
capability-graph artefact and no draft → ground → re-draft outer loop.

This module is a sanctioned ``import pydantic_ai`` site under
``agent/_pydanticai/`` (CLAUDE.md § Layer charters).
"""

from __future__ import annotations

from typing import cast

from pydantic_ai import Agent, models
from pydantic_ai.mcp import MCPToolset, StdioTransport

from molexp.agent.modes._planning import PlanGraph

__all__ = ["build_research_planner"]

# pydantic-ai's ``Agent(model=...)`` accepts any of these shapes.
type PydanticAiModel = models.Model | models.KnownModelName | str

_OUTPUT_RETRIES = 2
"""``Agent(output_retries=...)`` — pydantic-ai retries schema-parse
failures at the model level with the validation error fed back as a
short follow-up. Two retries is more than enough for a single structured
emit; the molexp side does no outer retry on schema_parse (see the
:mod:`molexp.agent._pydanticai.retry` policy)."""


_SYSTEM_PROMPT = (
    "You are a research-and-plan agent. You have an MCP toolset "
    "attached that exposes one or more project sources — use it to "
    "study what exists before you plan. The MCP catalog itself is your "
    "only source of truth about which packages exist; do not assume a "
    "fixed list of names from training data.\n"
    "\n"
    "WORKFLOW:\n"
    "  1. Read the catalog outlines exposed by the MCP server to "
    "discover what sources are available. Skim every returned module's "
    "summary; this is your map of what exists.\n"
    "  2. For EACH operation the intent requires, identify the "
    "primitives in the catalog that *together* implement it. You are "
    "looking for compositions of primitives, not single-API mappings. "
    "Building a complex object typically means combining a builder + "
    "templates + connectors + placers + sequence generators — NOT "
    "finding 'the one function that does it'. If a search returns no "
    "single match, that's expected; decompose the operation and find "
    "the primitives for each part.\n"
    "  3. Emit one typed PlanGraph. Each PlanStep carries:\n"
    "     - `api_refs`: every fully-qualified project symbol the step "
    "composes. Use the dotted qualname the MCP catalog returned — "
    "verbatim, never invented or paraphrased.\n"
    "     - `composition_notes`: 1-3 sentences explaining how the "
    "`api_refs` connect to implement this step. This is your own "
    "reasoning record; downstream codegen reads it.\n"
    "     - `io.inputs[*].name` and `io.outputs[*]`: name the values "
    "the step consumes / produces however reads naturally — domain "
    "terms (`peo_chain`, `forcefield`), filenames (`data.peo`), "
    "anything sensible. Downstream codegen sanitises these to Python "
    "locals automatically.\n"
    "     - `artifacts[*]`: when a step writes a side-effect file, "
    "list it here for downstream consumers that need the on-disk "
    "path rather than a Python value.\n"
    "\n"
    "CONSTRAINTS:\n"
    "  - Every PlanStep MUST have non-empty `api_refs`. A step with "
    "zero discovered primitives fails preflight.\n"
    "  - Every PlanStep MUST carry a `test_sketch` with "
    "`is_isolated_testable=true` and concrete synthetic inputs + "
    "assertions. A step is isolated-testable ONLY IF its isolated "
    "test fits in 1-3 SHAPE-LEVEL assertions (dict keys, value types, "
    "non-emptiness, simple counts). If you'd need content-level "
    "assertions — exact substrings of a generated script, specific "
    "numeric values, ordering of items in a long collection — the "
    "step is too big: SPLIT it into sub-steps where each sub-step's "
    "output IS the thing the content assertion would have checked. "
    "Example: instead of one 'write LAMMPS input script' step whose "
    "test would have to grep for ten substrings, emit "
    "'build init block', 'build force-field block', 'build run "
    "block', 'assemble script' as separate steps each returning the "
    "block as a string whose test asserts only `isinstance(out['x'], "
    "str) and out['x'].strip() != ''`.\n"
    "  - Steps form an acyclic graph. `depends_on` lists upstream step "
    "ids.\n"
    "  - Required outputs from the IntentSpec must appear verbatim as "
    "the `outputs` of some PlanStep (string-for-string match — the "
    "downstream preflight checks this).\n"
    "  - Write no executable code in the prose; only the typed plan "
    "structure.\n"
    "\n"
    "TOOLS — match by name pattern, not by exact identifier:\n"
    "  • CATALOG / OUTLINE (names contain 'outline', 'index', 'list', "
    "or 'tree') — hierarchical map of a source's packages → modules → "
    "symbols, each carrying a one-line summary. Read this first.\n"
    "  • CAPABILITY / SEARCH (names contain 'find', 'search', or "
    "'capability') — natural-language description → ranked real-source "
    "matches with qualname / kind / signature / summary.\n"
    "  • DETAIL / LOOKUP (names contain 'describe', 'get', or "
    "'inspect') — fully-qualified name → signature, docstring, source, "
    "relationships. Use to confirm a candidate symbol exists.\n"
)


def build_research_planner(
    model: PydanticAiModel,
    *,
    molmcp_command: str,
    molmcp_args: tuple[str, ...] = (),
    molmcp_env: dict[str, str] | None = None,
) -> Agent[None, PlanGraph]:
    """Construct the single MCP-attached research-and-plan agent.

    Args:
        model: pydantic-ai model identifier or instance.
        molmcp_command: Executable for the molmcp MCP server.
        molmcp_args: Optional CLI args for the MCP server.
        molmcp_env: Optional environment overlay for the MCP server.

    Returns:
        An ``Agent`` whose ``run(prompt)`` returns a typed
        :class:`PlanGraph`. pydantic-ai drives the model ↔ tool loop
        internally; no outer Python iteration is needed (or permitted —
        the rewrite spec's ac-005 invariant). Repair rewind context (the
        prior plan + failure description) is encoded by the calling
        stage in the ``user`` argument to ``agent.run(...)``, not here.
    """
    server = MCPToolset(
        StdioTransport(command=molmcp_command, args=list(molmcp_args), env=molmcp_env)
    )
    agent = Agent(
        model=model,
        output_type=PlanGraph,
        system_prompt=_SYSTEM_PROMPT,
        toolsets=[server],
        output_retries=_OUTPUT_RETRIES,
    )
    return cast("Agent[None, PlanGraph]", agent)
