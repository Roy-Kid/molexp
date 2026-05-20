"""Tests for the additive ``running`` / ``completed`` PlanState extension (ac-010).

Sub-spec 05 adds two members to :class:`PlanState` and four edges to
:data:`LEGAL_TRANSITIONS`. These tests pin the new states and edges and
assert every *other* edge out of ``running`` is rejected.
"""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import (
    LEGAL_TRANSITIONS,
    IllegalPlanTransitionError,
    PlanState,
    assert_legal_transition,
    is_legal_transition,
    legal_successors,
)

# ── new members exist ────────────────────────────────────────────────────


def test_planstate_has_running_and_completed() -> None:
    assert PlanState.running.value == "running"
    assert PlanState.completed.value == "completed"


def test_running_and_completed_in_transition_table() -> None:
    assert PlanState.running in LEGAL_TRANSITIONS
    assert PlanState.completed in LEGAL_TRANSITIONS


# ── the four new edges are legal ─────────────────────────────────────────


def test_ready_for_run_to_running_is_legal() -> None:
    assert_legal_transition(PlanState.ready_for_run, PlanState.running)
    assert is_legal_transition(PlanState.ready_for_run, PlanState.running)


def test_running_to_completed_is_legal() -> None:
    assert_legal_transition(PlanState.running, PlanState.completed)


def test_running_to_failed_is_legal() -> None:
    assert_legal_transition(PlanState.running, PlanState.failed)


def test_running_to_needs_clarification_is_legal() -> None:
    assert_legal_transition(PlanState.running, PlanState.needs_clarification)


def test_running_successors_are_exactly_the_three_new_targets() -> None:
    assert legal_successors(PlanState.running) == frozenset(
        {PlanState.completed, PlanState.failed, PlanState.needs_clarification}
    )


# ── completed is terminal ────────────────────────────────────────────────


def test_completed_is_terminal() -> None:
    assert legal_successors(PlanState.completed) == frozenset()


def test_completed_to_running_is_illegal() -> None:
    with pytest.raises(IllegalPlanTransitionError):
        assert_legal_transition(PlanState.completed, PlanState.running)


# ── every other edge out of running is rejected ──────────────────────────


@pytest.mark.parametrize(
    "dst",
    [
        state
        for state in PlanState
        if state not in {PlanState.completed, PlanState.failed, PlanState.needs_clarification}
    ],
)
def test_every_other_edge_out_of_running_is_illegal(dst: PlanState) -> None:
    with pytest.raises(IllegalPlanTransitionError):
        assert_legal_transition(PlanState.running, dst)


def test_ready_for_run_only_reaches_running() -> None:
    # ready_for_run was terminal before this sub-spec; now its sole
    # successor is running.
    assert legal_successors(PlanState.ready_for_run) == frozenset({PlanState.running})
    with pytest.raises(IllegalPlanTransitionError):
        assert_legal_transition(PlanState.ready_for_run, PlanState.completed)
