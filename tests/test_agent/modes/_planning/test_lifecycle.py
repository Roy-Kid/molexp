"""RED-phase tests for the lifecycle cluster of
``molexp.agent.modes._planning``.

The package does not exist yet; these tests fail at collection until the
implementation lands.

Covers, per the testing rules:

- Basics  — ``PlanState`` enum membership.
- Logic — the ``LEGAL_TRANSITIONS`` table covers every documented
  legal edge; ``is_legal_transition`` / ``legal_successors`` agree
  with it; a representative set of illegal pairs is rejected;
  ``assert_legal_transition`` raises ``IllegalPlanTransitionError``
  (a ``ValueError`` subclass) on an illegal pair and stays silent on
  a legal one.
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

# --------------------------------------------------------------------------
# the documented legal-edge table — single source of truth for the tests
# --------------------------------------------------------------------------

_EXPECTED_EDGES: dict[PlanState, set[PlanState]] = {
    PlanState.intake: {PlanState.needs_clarification, PlanState.exploring},
    PlanState.needs_clarification: {PlanState.exploring, PlanState.intake},
    PlanState.exploring: {PlanState.draft_plan, PlanState.needs_clarification},
    PlanState.draft_plan: {PlanState.preflight_failed, PlanState.awaiting_approval},
    PlanState.preflight_failed: {
        PlanState.exploring,
        PlanState.draft_plan,
        PlanState.failed,
    },
    PlanState.awaiting_approval: {
        PlanState.approved,
        PlanState.rejected,
        PlanState.draft_plan,
    },
    PlanState.approved: {PlanState.materializing},
    PlanState.materializing: {PlanState.validating, PlanState.failed},
    PlanState.validating: {
        PlanState.ready_for_run,
        PlanState.draft_plan,
        PlanState.failed,
    },
    PlanState.ready_for_run: set(),
    PlanState.rejected: set(),
    PlanState.failed: set(),
}


# --------------------------------------------------------------------------
# basics — PlanState enum
# --------------------------------------------------------------------------


def test_plan_state_has_all_twelve_members() -> None:
    assert {m.value for m in PlanState} == {
        "intake",
        "needs_clarification",
        "exploring",
        "draft_plan",
        "preflight_failed",
        "awaiting_approval",
        "approved",
        "materializing",
        "validating",
        "ready_for_run",
        "rejected",
        "failed",
    }


def test_plan_state_string_value() -> None:
    assert PlanState.ready_for_run == "ready_for_run"


# --------------------------------------------------------------------------
# logic — LEGAL_TRANSITIONS table shape
# --------------------------------------------------------------------------


def test_legal_transitions_covers_every_state() -> None:
    assert set(LEGAL_TRANSITIONS) == set(PlanState)


def test_legal_transitions_matches_documented_table() -> None:
    for src, expected in _EXPECTED_EDGES.items():
        assert LEGAL_TRANSITIONS[src] == frozenset(expected), src


def test_legal_transitions_values_are_frozensets() -> None:
    for successors in LEGAL_TRANSITIONS.values():
        assert isinstance(successors, frozenset)


@pytest.mark.parametrize(
    "terminal",
    [PlanState.ready_for_run, PlanState.rejected, PlanState.failed],
)
def test_terminal_states_have_no_successors(terminal: PlanState) -> None:
    assert LEGAL_TRANSITIONS[terminal] == frozenset()


# --------------------------------------------------------------------------
# logic — is_legal_transition covers every legal edge
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst"),
    [(src, dst) for src, dsts in _EXPECTED_EDGES.items() for dst in dsts],
)
def test_is_legal_transition_true_for_every_legal_edge(src: PlanState, dst: PlanState) -> None:
    assert is_legal_transition(src, dst) is True


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (PlanState.intake, PlanState.approved),
        (PlanState.intake, PlanState.ready_for_run),
        (PlanState.exploring, PlanState.approved),
        (PlanState.draft_plan, PlanState.materializing),
        (PlanState.approved, PlanState.ready_for_run),
        (PlanState.awaiting_approval, PlanState.materializing),
        (PlanState.ready_for_run, PlanState.draft_plan),
        (PlanState.rejected, PlanState.intake),
        (PlanState.failed, PlanState.exploring),
        (PlanState.materializing, PlanState.ready_for_run),
    ],
)
def test_is_legal_transition_false_for_illegal_pairs(src: PlanState, dst: PlanState) -> None:
    assert is_legal_transition(src, dst) is False


# --------------------------------------------------------------------------
# logic — legal_successors agrees with the table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src", list(PlanState))
def test_legal_successors_matches_table(src: PlanState) -> None:
    assert legal_successors(src) == LEGAL_TRANSITIONS[src]


def test_legal_successors_returns_frozenset() -> None:
    assert isinstance(legal_successors(PlanState.intake), frozenset)


# --------------------------------------------------------------------------
# logic — assert_legal_transition
# --------------------------------------------------------------------------


def test_assert_legal_transition_silent_on_legal_pair() -> None:
    # Must not raise.
    assert assert_legal_transition(PlanState.intake, PlanState.exploring) is None


def test_assert_legal_transition_raises_on_illegal_pair() -> None:
    with pytest.raises(IllegalPlanTransitionError):
        assert_legal_transition(PlanState.intake, PlanState.approved)


def test_illegal_plan_transition_error_is_value_error_subclass() -> None:
    assert issubclass(IllegalPlanTransitionError, ValueError)


def test_assert_legal_transition_illegal_pair_caught_as_value_error() -> None:
    with pytest.raises(ValueError):
        assert_legal_transition(PlanState.failed, PlanState.exploring)
