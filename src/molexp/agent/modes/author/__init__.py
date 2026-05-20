"""AuthorMode — materialize an approved typed plan into a workspace (sub-spec 04).

AuthorMode consumes the
:class:`~molexp.agent.modes.plan.handoff.ApprovedPlanHandoff` PlanMode
emits and turns the typed :class:`~molexp.agent.modes._planning.PlanGraph`
into a materialized, validated experiment workspace: the lowered
workflow IR, generated per-task source + tests, a package skeleton, and
a manifest. It runs each generated task's test through an
isolated-subprocess debug loop, gates file generation behind the
``approve_materialization`` human approval, and emits a
:class:`MaterializedWorkspaceHandoff` RunMode (sub-spec 05) consumes.

Public surface:

- :class:`AuthorMode` / :class:`AuthorModeConfig` — the harness-based mode.
- :class:`MaterializedWorkspaceHandoff` — AuthorMode's terminal output.
"""

from __future__ import annotations

from molexp.agent.modes.author._mode import AuthorMode, AuthorModeConfig
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff

__all__ = [
    "AuthorMode",
    "AuthorModeConfig",
    "MaterializedWorkspaceHandoff",
]
