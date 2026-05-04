"""Native tool collection (spec §3, §6.2).

Per Decision T1 (spec §13) the package is import-only: each tool here
is decorated with :func:`native_tool` to attach a :class:`ToolSpec`,
but registration is deferred until :class:`AgentService` walks the
package on construction. There is no module-level singleton.

Submodules:

- :mod:`workspace` — list/create projects, experiments, runs.
- :mod:`workflow` — run lifecycle + workflow-IR binding (mutating tools
  default to ``mutates=True`` so plan mode hides them).
- :mod:`chat` — :func:`ask_user`.

The legacy ``exit_plan_mode`` tool is intentionally absent: per
Decision O2 plan mode is a runner-side state machine, not a tool.
"""

from molexp.agent.tools.native import chat, workflow, workspace

__all__ = ["chat", "workflow", "workspace"]
