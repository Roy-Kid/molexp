"""Native ``/plan`` skill.

Plan mode keeps the full native tool surface; the constraint is on
the output shape (a structured plan + workflow IR handoff). The
instructions text is :data:`PLAN_MODE_ADDENDUM` so the prompt composer
reads it directly off the skill rather than hardcoding it.
"""

from __future__ import annotations

from molexp.agent.context.prompt import PLAN_MODE_ADDENDUM
from molexp.agent.persistence import Scope
from molexp.agent.skills.store import SkillStore
from molexp.agent.skills.types import Skill

PLAN_SKILL = Skill(
    id="native-plan",
    name="Plan mode",
    description=(
        "Plan mode. The agent explores the workspace, drafts a "
        "structured execution plan + a molexp workflow IR, and hands "
        "the bundle back for explicit approval. Tools are NOT "
        "restricted — the constraint is on the output shape, not the "
        "surface."
    ),
    goal_template="",
    slash_name="",
    instructions=PLAN_MODE_ADDENDUM,
    default_plan_mode=True,
    tags=["native", "mode"],
    scope=Scope.NATIVE,
    created_at="",
    updated_at="",
)

SkillStore.register(PLAN_SKILL)


__all__ = ["PLAN_SKILL"]
