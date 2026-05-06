"""PromptComposer: deterministic layered system prompt assembly.

Layer order:

1. Base prompt (built into the harness or supplied by config).
2. Workspace addendum (workspace-level instructions).
3. Skill addendum (re-resolved every turn from ``SkillStore``).
4. Per-session ``instructions_override`` (replaces 1-3 entirely when
   set).
5. Plan-mode addendum (always tailed when ``plan_mode=True``).

Each non-empty section is wrapped in a deterministic header so
prompts are stable across runs and trivially diffable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

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

Plan mode does NOT restrict your tool surface — you may freely call
``list_task_types``, ``list_workflow_templates``, ``list_projects``,
inspect runs, read assets, etc. as needed to compose a competent plan.
The constraint is on your **output**: this turn ends with a plan
emission. The harness routes the structured plan preview to the
user's UI for approve / reject; on rejection the user feedback
returns as a synthetic user message.

## Plan output

Emit two things this turn:

1. A numbered, human-readable plan (markdown).
2. Optionally, a fenced ``json`` block carrying any of:
   - ``python_script``: a Python molexp script rendering the plan,
     suitable for the user to read or edit.
   - ``mermaid``: a Mermaid flowchart of the plan's task graph.
   - ``intervention_points``: a list of names for explicit user-input
     anchors in the plan.

These three fields populate the derived ``WorkflowPreviewView`` that
the UI surfaces alongside the markdown. They are advisory; the plan
itself is the source of truth.

On approval the session flips out of plan mode and you proceed to
bind / execute the (possibly user-edited) plan. On rejection you
receive the user's feedback as a synthetic user message and should
revise + emit the plan again.\
"""


class PromptLayer(BaseModel):
    """One labeled section of the assembled system prompt."""

    model_config = ConfigDict(frozen=True)

    title: str
    body: str


class PromptComposer:
    """Compose a system prompt from layered sections."""

    SECTION_HEADER = "## {title}"

    def compose(
        self,
        *,
        base: str,
        workspace: str = "",
        skill: str = "",
        override: str | None = None,
    ) -> str:
        """Return the rendered system prompt.

        ``override`` short-circuits all layering when set; the harness
        stores it as a flat string and the composer never merges it
        with other sources.
        """

        if override is not None:
            return override.strip()

        layers = [
            PromptLayer(title="Base", body=base),
            PromptLayer(title="Workspace", body=workspace),
            PromptLayer(title="Skill", body=skill),
        ]
        sections = [self._render(layer) for layer in layers if layer.body.strip()]
        return "\n\n".join(sections).strip()

    def _render(self, layer: PromptLayer) -> str:
        header = self.SECTION_HEADER.format(title=layer.title)
        return f"{header}\n{layer.body.strip()}"


def compose_system_prompt(
    *,
    base: str = BASE_SYSTEM_PROMPT,
    workspace_instructions: str = "",
    skill_instructions: str = "",
    session_override: str | None = None,
    plan_mode: bool = False,
) -> str:
    """Compose the final system prompt sent to the model.

    ``session_override`` of ``None`` (default) means "use layered
    composition"; an empty string means "blank session prompt" (still
    tailed by the plan-mode addendum when applicable).
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


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "PLAN_MODE_ADDENDUM",
    "PromptComposer",
    "PromptLayer",
    "compose_system_prompt",
]
