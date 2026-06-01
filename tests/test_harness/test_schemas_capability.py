"""Tests for ToolCapability (Phase 4 §5.2 schema).

Locks the wire format:
- frozen pydantic round-trip
- 13 fields per harness-goal.md §5.2
- defaults for optional fields
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _required_kwargs() -> dict:
    return {
        "id": "molpy.builder.polymer.GBigSmilesCompiler",
        "package": "molpy",
        "name": "GBigSmilesCompiler",
        "description": "Compile a Generalized BigSMILES expression into an atomistic configuration",
        "input_schema": {
            "type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"],
        },
        "output_schema": {"type": "object", "properties": {"structure": {"type": "string"}}},
    }


def test_tool_capability_required_only_round_trip() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    cap = ToolCapability(**_required_kwargs())
    dumped = cap.model_dump_json()
    rehydrated = ToolCapability.model_validate_json(dumped)
    assert rehydrated == cap


def test_tool_capability_full_round_trip() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    cap = ToolCapability(
        **_required_kwargs(),
        callable_path="molpy.builder.polymer.GBigSmilesCompiler",
        cli_template=["molpy", "build", "{smiles}"],
        side_effects=["fs_write"],
        supported_backends=["local", "slurm"],
        examples=[{"smiles": "{[<]CCO[>]}", "structure": "out.pdb"}],
        version="0.1.0",
        tags=["polymer", "builder"],
    )
    dumped = cap.model_dump_json()
    rehydrated = ToolCapability.model_validate_json(dumped)
    assert rehydrated == cap


def test_tool_capability_defaults() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    cap = ToolCapability(**_required_kwargs())
    assert cap.callable_path is None
    assert cap.cli_template is None
    assert cap.side_effects == []
    assert cap.supported_backends == ["local"]
    assert cap.examples == []
    assert cap.version is None
    assert cap.tags == []


def test_tool_capability_is_frozen() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    cap = ToolCapability(**_required_kwargs())
    with pytest.raises(ValidationError):
        cap.id = "mutated"  # type: ignore[misc]


def test_tool_capability_missing_required_raises() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    with pytest.raises(ValidationError):
        ToolCapability(  # type: ignore[call-arg]
            id="x",
            package="y",
            name="z",
            description="d",
            # missing input_schema, output_schema
        )


def test_tool_capability_default_factories_are_independent() -> None:
    from molexp.harness.schemas.capability import ToolCapability

    a = ToolCapability(**_required_kwargs())
    b = ToolCapability(**_required_kwargs())
    assert a.side_effects is not b.side_effects
    assert a.tags is not b.tags
    assert a.examples is not b.examples


def test_tool_capability_re_exported_from_schemas() -> None:
    from molexp.harness.schemas import ToolCapability as via_schemas
    from molexp.harness.schemas.capability import ToolCapability as via_module

    assert via_schemas is via_module


def test_tool_capability_re_exported_from_top_level() -> None:
    from molexp.harness import ToolCapability as via_top
    from molexp.harness.schemas.capability import ToolCapability as via_module

    assert via_top is via_module
