"""Public-surface contract for ``molexp.agent`` (spec ac-001 / ac-002 / ac-003 / ac-006).

The original "four-name contract" covered the mode-orchestration core
(:class:`AgentRunner`, :class:`AgentMode`, :class:`AgentRunResult`,
:class:`AgentSession`). It has been extended with workflow-orthogonal
policy primitives — :class:`GatePolicy`, :class:`AutoApproveGatePolicy`,
:func:`static_gate_policy_lookup` — because policies are not
mode-specific and any workflow-bearing mode consumes them. See
``feedback_policy_at_agent_layer.md``. Concrete *interactive* gates
(e.g. :class:`~molexp.agent.modes.plan.PromptGatePolicy`) ship under
their owning mode's subpackage and are not part of the cross-mode
surface.
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
        # Workflow-orthogonal policies (parallel to mode).
        "GatePolicy",
        "AutoApproveGatePolicy",
        "static_gate_policy_lookup",
    }


def test_modes_all_is_the_three_pair_contract() -> None:
    assert set(modes.__all__) == {
        "PlanMode",
        "PlanModeConfig",
        "PlanResult",
        "ChatMode",
        "ChatModeConfig",
        "ReviewMode",
        "ReviewModeConfig",
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
