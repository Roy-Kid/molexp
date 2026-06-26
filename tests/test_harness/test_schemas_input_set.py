"""Tests for InputSet — the declarative parameter-space spec (step 6).

The harness InputSet only *describes* the sweep; cell expansion is delegated
to the workspace ParamSpace family. This locks the wire format + confirms a
grid InputSet bridges cleanly into workspace.GridSpace.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _input_set_kwargs() -> dict:
    from molexp.harness.schemas.input_set import SweepAxis

    return {
        "id": "is-1",
        "experiment_spec_id": "spec-1",
        "title": "field x temperature sweep",
        "sweep_axes": [
            SweepAxis(name="E_field", values=[1.0e6, 2.0e6], source="agent_inferred"),
            SweepAxis(name="temperature", values=[280, 300, 320], source="user_provided"),
        ],
        "strategy": "grid",
        "total_runs": 6,
    }


def test_input_set_full_round_trip() -> None:
    from molexp.harness.schemas.input_set import InputSet

    iset = InputSet(**_input_set_kwargs())
    assert InputSet.model_validate_json(iset.model_dump_json()) == iset
    assert iset.total_runs == 6
    assert iset.sweep_axes[0].name == "E_field"


def test_input_set_defaults() -> None:
    from molexp.harness.schemas.input_set import InputSet

    iset = InputSet(id="is-1", experiment_spec_id="spec-1", title="t")
    assert iset.sweep_axes == []
    assert iset.strategy == "grid"
    assert iset.total_runs == 1
    assert iset.random_seed is None


def test_input_set_bridges_to_workspace_gridspace() -> None:
    """A grid InputSet maps to a workspace GridSpace whose count == total_runs."""
    from molexp.harness.schemas.input_set import InputSet
    from molexp.workspace.param import GridSpace

    iset = InputSet(**_input_set_kwargs())
    grid = GridSpace({axis.name: list(axis.values) for axis in iset.sweep_axes})
    assert grid.count() == iset.total_runs


def test_input_set_is_frozen() -> None:
    from molexp.harness.schemas.input_set import InputSet

    iset = InputSet(**_input_set_kwargs())
    with pytest.raises(ValidationError):
        iset.total_runs = 99  # type: ignore[misc]


def test_input_set_rejects_unknown_strategy() -> None:
    from molexp.harness.schemas.input_set import InputSet

    with pytest.raises(ValidationError):
        InputSet(id="is-1", experiment_spec_id="spec-1", title="t", strategy="bogus")  # type: ignore[arg-type]
