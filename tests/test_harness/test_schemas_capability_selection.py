"""Tests for CapabilitySelection — the LLM's pick of needed capabilities (step 3).

Locks the wire format the ``capability_selector`` agent returns: a short list of
``capability_id``\\ s, each with a one-line reason, plus optional notes. Frozen
pydantic round-trip + json-schema shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _selection_kwargs() -> dict:
    from molexp.harness.schemas.capability_selection import SelectedCapability

    return {
        "selected": [
            SelectedCapability(id="molpy.core.cg.CoarseGrain", reason="builds the CG beads"),
            SelectedCapability(
                id="molpy.io.writers.write_lammps_data", reason="writes the data file"
            ),
        ],
        "notes": "no packing capability available",
    }


def test_capability_selection_full_round_trip() -> None:
    from molexp.harness.schemas.capability_selection import CapabilitySelection

    sel = CapabilitySelection(**_selection_kwargs())
    rehydrated = CapabilitySelection.model_validate_json(sel.model_dump_json())
    assert rehydrated == sel
    assert sel.selected[0].id == "molpy.core.cg.CoarseGrain"
    assert sel.selected[0].reason == "builds the CG beads"
    assert sel.notes == "no packing capability available"


def test_capability_selection_defaults_empty() -> None:
    from molexp.harness.schemas.capability_selection import CapabilitySelection

    sel = CapabilitySelection()
    assert sel.selected == []
    assert sel.notes == ""


def test_capability_selection_is_frozen() -> None:
    from molexp.harness.schemas.capability_selection import CapabilitySelection

    sel = CapabilitySelection(**_selection_kwargs())
    with pytest.raises(ValidationError):
        sel.notes = "mutated"  # type: ignore[misc]


def test_selected_capability_requires_id_and_reason() -> None:
    from molexp.harness.schemas.capability_selection import SelectedCapability

    with pytest.raises(ValidationError):
        SelectedCapability(id="molpy.core.cg.CoarseGrain")  # type: ignore[call-arg]


def test_capability_selection_model_json_schema_is_dict() -> None:
    from molexp.harness.schemas.capability_selection import CapabilitySelection

    schema = CapabilitySelection.model_json_schema()
    assert isinstance(schema, dict)
    assert "selected" in schema.get("properties", {})
