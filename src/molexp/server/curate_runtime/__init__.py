"""Server-tier workspace-curation runtime.

The single shared backend path that turns a natural-language curation request
into an in-process workspace mutation, plus the background-task + gateway
plumbing both ``molexp curate`` (CLI) and the ``curate-tasks`` route delegate to.

- :func:`run_curation_flow` — the ONE code path (discover → plan → gate → invoke
  in-process), mirroring the ``materialize_plan_records`` precedent.
- :class:`CurationInvocation` / :class:`CurationResult` — the planner's structured
  output and the flow's outcome.
- :func:`resolve_curation_arguments` — reconstructs live-object arguments from the
  planner's JSON references.
- :class:`CurateTask` / :class:`CurateTaskRegistry` — the background-task wrapper.
- :func:`build_curate_gateway` (+ factory seam) — the production gateway builder.
"""

from __future__ import annotations

from molexp.server.curate_runtime.flow import (
    CurationArgumentError,
    CurationInvocation,
    CurationResult,
    resolve_curation_arguments,
    run_curation_flow,
)
from molexp.server.curate_runtime.gateway import (
    build_curate_gateway,
    curate_agent_responses,
    curate_output_kinds,
    curate_system_prompts,
    reset_curate_gateway_factory,
    set_curate_gateway_factory,
)
from molexp.server.curate_runtime.task import (
    CurateTask,
    CurateTaskRegistry,
    CurateTaskStatus,
)

__all__ = [
    "CurateTask",
    "CurateTaskRegistry",
    "CurateTaskStatus",
    "CurationArgumentError",
    "CurationInvocation",
    "CurationResult",
    "build_curate_gateway",
    "curate_agent_responses",
    "curate_output_kinds",
    "curate_system_prompts",
    "reset_curate_gateway_factory",
    "resolve_curation_arguments",
    "run_curation_flow",
    "set_curate_gateway_factory",
]
