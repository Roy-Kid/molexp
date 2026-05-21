"""Public-surface contract for ``molexp.agent`` (spec ac-001 / ac-002 / ac-003 / ac-006).

The original "four-name contract" covered the mode-orchestration core
(:class:`AgentRunner`, :class:`AgentMode`, :class:`AgentRunResult`,
:class:`AgentSession`). It is extended with three workflow-orthogonal
approval primitives — :class:`ReviewDecision`, the :data:`ReviewPolicy`
callable alias, and the bundled :func:`cli_ask` policy — because the
``before_approval`` hook is not mode-specific and any mode that reaches
an :class:`ApprovalGate` consults it.
"""

from __future__ import annotations

import inspect

import molexp.agent as agent
import molexp.agent.modes as modes


def test_agent_all_is_the_public_contract() -> None:
    assert set(agent.__all__) == {
        # Mode orchestration core.
        "AgentRunner",
        "AgentMode",
        "AgentRunResult",
        "AgentSession",
        # Workflow-orthogonal approval primitives (parallel to mode).
        "ReviewPolicy",
        "ReviewDecision",
        "cli_ask",
    }


def test_modes_all_is_the_five_mode_contract() -> None:
    """``ChatMode`` / ``PlanMode`` / ``AuthorMode`` / ``RunMode`` /
    ``ReviewMode`` — all five modes ship on the harness."""
    assert set(modes.__all__) == {
        "ApprovedPlanHandoff",
        "AuthorMode",
        "AuthorModeConfig",
        "ChatMode",
        "ChatModeConfig",
        "MaterializedWorkspaceHandoff",
        "PlanFolder",
        "PlanMode",
        "PlanModeConfig",
        "RepairEscalation",
        "ReviewMode",
        "ReviewModeConfig",
        "RunFolder",
        "RunMode",
        "RunModeConfig",
        "RunProgress",
        "RunReport",
        "StepProgress",
    }


def test_no_pydantic_ai_or_pydantic_graph_in_public_surface() -> None:
    bad = [
        n
        for n in (*agent.__all__, *modes.__all__)
        if n.startswith(("pydantic_ai", "pydantic_graph"))
    ]
    assert not bad, bad


def test_no_factory_callables_in_public_surface() -> None:
    """Anti-factory invariant — ac-006."""
    for ns in (agent, modes):
        for name, obj in inspect.getmembers(ns):
            if name in ns.__all__ and callable(obj):
                assert not name.startswith(("create_", "build_", "make_")), name
