"""Package-builtin skills — the highest-precedence tier for skill discovery.

These skills are registered in code (not on disk) and ship with molexp.
They cannot be deleted from the UI; users may shadow a builtin's
``slash_name`` with a same-named workspace or user-home skill, in which
case the UI shows a "shadowing builtin" indicator with a "reset" affordance.

The current set:

- ``builtin-plan`` — the ``/plan`` mode. Restricts the catalog to
  read-only inspection tools plus :func:`exit_plan_mode`, injects the
  plan-mode addendum into the system prompt, and forces the agent to
  emit its plan via the structured ``exit_plan_mode`` tool rather than
  free-form prose.
"""

from __future__ import annotations

from .skills import Skill, SkillScope


def _build_plan_skill() -> Skill:
    """Construct the builtin ``/plan`` skill.

    Plan mode in molexp is *not* a tool-restriction mode — the agent has
    the full native tool surface so it can call ``list_task_types``,
    explore the workspace, even probe MCP servers as needed. The
    constraint lives in the **output contract**: the agent must finalize
    via :func:`exit_plan_mode` and submit a structured plan + a valid
    molexp workflow IR. The user reviews the plan + a rendered task
    graph in the chat, edits the IR if needed, and approves; the same
    session then resumes with ``plan_mode=False`` and executes.

    The instructions text is the canonical PLAN_MODE_ADDENDUM — kept in
    a single place so the system-prompt composer reads it directly off
    the skill instead of hard-coding it.
    """
    from ._pydantic_ai.system_prompt import PLAN_MODE_ADDENDUM

    return Skill(
        id="builtin-plan",
        name="Plan mode",
        description=(
            "Plan mode. The agent explores the workspace, drafts a "
            "structured execution plan + a molexp workflow IR, and hands "
            "the bundle back via exit_plan_mode for explicit approval. "
            "Tools are NOT restricted — the constraint is on the output "
            "shape, not the surface."
        ),
        goal_template="",  # plan mode is launched from a regular goal + flag
        slash_name="",  # /plan is reserved at the chat layer; lookup is by ID
        instructions=PLAN_MODE_ADDENDUM,
        default_plan_mode=True,
        allowed_tools=[],  # no restriction — agent has full tool surface
        denied_tools=[],
        requires_exit_tool="exit_plan_mode",
        builtin=True,
        scope=SkillScope.BUILTIN,
        tags=["builtin", "mode"],
    )


_BUILTIN_SKILLS: list[Skill] | None = None


def list_builtin_skills() -> list[Skill]:
    """Return every package-builtin skill, lazily constructed.

    Lazy construction avoids an import cycle with
    :mod:`._pydantic_ai.system_prompt` at module import time.
    """
    global _BUILTIN_SKILLS
    if _BUILTIN_SKILLS is None:
        _BUILTIN_SKILLS = [_build_plan_skill()]
    return list(_BUILTIN_SKILLS)


def get_builtin_skill(skill_id: str) -> Skill | None:
    """Look up a builtin skill by its stable ID (e.g. ``"builtin-plan"``)."""
    for skill in list_builtin_skills():
        if skill.id == skill_id:
            return skill
    return None
