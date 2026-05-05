"""Native tool collection.

Each tool here is decorated with :func:`native_tool` to attach a
:class:`ToolSpec`. Registration is deferred until
:class:`AgentService` walks the package on construction — there is no
module-level singleton.

Submodules:

- :mod:`workspace` — list/create projects, experiments, runs.
- :mod:`workflow` — run lifecycle + workflow-IR binding (mutating
  tools default to ``mutates=True`` so plan mode hides them).
- :mod:`chat` — :func:`ask_user`.
- :mod:`web` — Brave Search API web lookup (:func:`web_search`).

Plan mode is a runner-side state machine, not a tool, so there is no
``exit_plan_mode`` here; reject feedback flows back as a synthetic
user message instead.
"""

from molexp.agent.tools.native import chat, web, workflow, workspace

__all__ = ["chat", "web", "workflow", "workspace"]
