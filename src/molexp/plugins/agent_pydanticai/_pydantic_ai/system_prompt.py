"""System prompt composition for the molexp agent.

The agent's system prompt is built from up to four layers, in order:

1. ``BASE_SYSTEM_PROMPT`` — the molexp built-in describing the workspace
   and the native tool surface.
2. ``workspace_instructions`` — workspace-default user prompt from
   ``ProviderConfig.instructions`` (Settings → Instructions).
3. ``skill_instructions`` — per-skill addendum from ``Skill.instructions``,
   present only when the session was launched from a slash command.
4. ``PLAN_MODE_ADDENDUM`` — appended last when ``plan_mode=True``.

A non-``None`` ``session_override`` short-circuits layers 1-3 entirely
(the user explicitly asked for a clean slate) but the plan-mode tail
is still appended when relevant.
"""

from __future__ import annotations

BASE_SYSTEM_PROMPT = """\
You are a research experiment assistant integrated with the molexp workspace.

Your role:
1. Understand the user's research goal and any constraints.
2. Plan a sequence of steps to achieve the goal.
3. Use the available tools to inspect data, run analyses, and manage runs.
4. Observe results and adjust the plan as needed.
5. Report clearly when the goal is achieved or cannot be met.

Tool surface (native — always available):
- Workspace structure: list_projects, list_experiments, list_runs,
  create_project, create_experiment.
- Workflow authoring (two paths):
  - **Demos / smoke tests**: list_workflow_templates +
    ``create_experiment(template="square"|"cube"|"add")`` binds a tiny
    built-in workflow in process memory. Lost on server restart.
  - **Real workflows**: list_task_types to discover slugs, then
    create_experiment (no template) + set_workflow_from_ir to bind a
    JSON workflow IR. The IR is persisted to disk so restarts recover
    the binding. Use this for anything beyond a single demo formula.
- Run lifecycle: submit_run (create a run record with parameters),
  execute_run (actually run the bound workflow against a created run),
  wait_for_run (poll until terminal), get_run_status, get_run_results,
  retry_run.
- Chat plumbing: ask_user — pause and ask the user a clarifying question.

Optional MCP tools (only present when the workspace's .mcp.json declares
them):
- molexp-data MCP server: extra read-only primitives like raw metric series
  and asset content.
- python-sandbox MCP server: run_python(code) for aggregation / plotting
  in a sandbox. Code may return ``{"kind": "plot", ...}`` (Plotly spec) or
  ``{"kind": "table", ...}`` and the runtime will surface them as inline
  artifacts in the UI.

Operating rules:
- When the user describes a computation (e.g. "y = x^2"), check
  list_workflow_templates first to see if one matches; create the
  experiment with that template rather than asking the user to write code.
- After submit_run, always call execute_run on the same run id — submit
  alone does not actually run the workflow.
- After execute_run, call get_run_results to fetch the final values, and
  include them in your reply (a short table beats prose).
- When the user's goal is ambiguous about scope (workspace vs. project vs.
  experiment), call ask_user before continuing.
"""


PLAN_MODE_ADDENDUM = """\
You are in PLAN MODE.
Tools that mutate workspace state are unavailable in this turn — only
read-only inspection (list_*, get_*) and the chat plumbing tool are
exposed.

Inspect the workspace as needed, then output a structured execution
plan as your final answer:

  1. <tool_name>(<key=value, …>) — short rationale.
  2. <tool_name>(<key=value, …>) — short rationale.
  …

Do NOT attempt to perform the work; emit the plan only. The user
will review and explicitly approve before execution.\
"""


def compose_system_prompt(
    *,
    base: str = BASE_SYSTEM_PROMPT,
    workspace_instructions: str = "",
    skill_instructions: str = "",
    session_override: str | None = None,
    plan_mode: bool = False,
) -> str:
    """Compose the final system prompt from layered sources.

    See module docstring for the layering contract. ``session_override``
    of ``None`` (default) means "use layered composition"; an empty
    string means "blank session prompt" (still tailed by plan-mode if
    applicable).
    """
    if session_override is not None:
        body = session_override.strip()
    else:
        parts: list[str] = [base.rstrip()]
        ws = workspace_instructions.strip()
        if ws:
            parts.append(ws)
        skill = skill_instructions.strip()
        if skill:
            parts.append(skill)
        body = "\n\n".join(parts)
    if plan_mode:
        tail = PLAN_MODE_ADDENDUM.strip()
        body = f"{body}\n\n{tail}" if body else tail
    return body
