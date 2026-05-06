"""Budgets, workspace boundaries, side-effect constraints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConstraintSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Hard caps enforced by the orchestration layer.

    Soft hints to the model travel as :class:`ModelBudget`; this set
    is the harness-side enforcement surface so a model can never
    overshoot through SDK convention drift.
    """

    max_turns: int = 50
    max_total_input_tokens: int = 1_000_000
    max_total_output_tokens: int = 1_000_000
    max_tool_calls: int = 200
    max_tool_calls_per_turn: int = 16
    allow_writes_outside_run: bool = False
