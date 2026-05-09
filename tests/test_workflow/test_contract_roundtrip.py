"""Round-trip tests for the new contract / YAML compiler surface.

Covers:

- ``contract_to_dict`` ⇄ ``dict_to_contract`` (field-equal).
- ``ir_to_yaml`` ⇄ ``yaml_to_ir`` (text round-trip via JSON-shaped dict).
- Full chain: ``WorkflowContract`` → ``contract_to_dict`` → YAML →
  ``yaml_to_ir`` → ``dict_to_contract`` is field-equal.
- Back-compat: an old IR JSON without ``workflow_contract`` survives
  ``WorkflowSpec.from_dict`` → ``spec_to_ir`` (no contract injected).
- Safety: ``yaml_to_ir`` rejects unsafe YAML tags and non-dict tops.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from molexp.workflow.compiler import default_compiler
from molexp.workflow.contract import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
    ValidationCheck,
    ValidationCheckId,
    WorkflowContract,
)


def _sample_contract() -> WorkflowContract:
    return WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
                artifacts=(
                    ArtifactDecl(
                        path="artifacts/a.json",
                        mime="application/json",
                        produced_by="A",
                    ),
                ),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="x", type="int", source="A", description="from A"),),
            ),
        ),
        validation_checks=(
            ValidationCheck(
                id=ValidationCheckId.outputs_match_downstream_inputs,
                severity="error",
            ),
        ),
    )


# ── contract_to_dict ⇄ dict_to_contract ────────────────────────────────────


def test_contract_dict_roundtrip_is_field_equal() -> None:
    contract = _sample_contract()
    dumped = default_compiler.contract_to_dict(contract)
    rebuilt = default_compiler.dict_to_contract(dumped)
    assert rebuilt == contract


def test_dict_to_contract_rejects_unknown_top_level_key() -> None:
    dumped = default_compiler.contract_to_dict(_sample_contract())
    dumped["stray"] = 1
    with pytest.raises(ValidationError):
        default_compiler.dict_to_contract(dumped)


# ── ir_to_yaml ⇄ yaml_to_ir ────────────────────────────────────────────────


def test_yaml_text_roundtrips_through_dict() -> None:
    contract = _sample_contract()
    dumped = default_compiler.contract_to_dict(contract)
    text = default_compiler.ir_to_yaml(dumped)
    parsed = default_compiler.yaml_to_ir(text)
    assert parsed == dumped


def test_full_contract_yaml_chain_is_field_equal() -> None:
    """contract → dict → YAML text → dict → contract — field-equal."""
    contract = _sample_contract()
    text = default_compiler.ir_to_yaml(default_compiler.contract_to_dict(contract))
    rebuilt = default_compiler.dict_to_contract(default_compiler.yaml_to_ir(text))
    assert rebuilt == contract


# ── YAML safety ────────────────────────────────────────────────────────────


def test_yaml_to_ir_rejects_python_object_tag() -> None:
    """``!!python/object`` is rejected by ``safe_load`` — we forward
    that error rather than constructing an arbitrary object."""
    unsafe = "!!python/object/apply:os.system [echo hello]\n"
    with pytest.raises(yaml.YAMLError):
        default_compiler.yaml_to_ir(unsafe)


def test_yaml_to_ir_rejects_non_dict_root() -> None:
    """A list-rooted YAML doc isn't an IR shape; the loader rejects it."""
    with pytest.raises(ValueError):
        default_compiler.yaml_to_ir("- a\n- b\n")


def test_yaml_to_ir_rejects_scalar_root() -> None:
    with pytest.raises(ValueError):
        default_compiler.yaml_to_ir("just-a-string\n")


# ── Back-compat: old workflow IR (no contract) round-trips intact ──────────


def test_old_workflow_ir_without_contract_round_trips_without_contract() -> None:
    """Loading an IR JSON dict that has no ``workflow_contract`` key
    via :meth:`Workflow.from_dict`, then re-serializing via
    :meth:`Workflow.to_dict`, must not inject a ``workflow_contract``
    section."""
    from molexp.workflow.registry import default_registry
    from molexp.workflow.task import Task

    class Echo(Task):
        async def execute(self, ctx):  # type: ignore[no-untyped-def, override]
            return None

    if not default_registry.has("test.echo_back_compat"):
        default_registry.register("test.echo_back_compat", lambda _config: Echo())

    ir_in = {
        "workflow_id": "workflow_abc12345",
        "name": "back_compat",
        "task_configs": [
            {
                "task_id": "k",
                "task_type": "test.echo_back_compat",
                "config": {},
                "status": "pending",
            },
        ],
        "links": [],
        "metadata": {
            "label": None,
            "description": None,
            "tags": [],
            "custom": {},
        },
    }
    spec = default_compiler.ir_to_spec(ir_in)
    ir_out = default_compiler.spec_to_ir(spec)
    assert "workflow_contract" not in ir_out


# ── Spec ↔ YAML composition ────────────────────────────────────────────────


def test_spec_yaml_round_trip_via_compiler() -> None:
    """``spec_to_yaml`` ⇄ ``yaml_to_spec`` is byte-stable through the
    JSON IR (slugged-task workflows only, matching the existing
    JSON-IR convention)."""
    from molexp.workflow.registry import default_registry
    from molexp.workflow.spec import WorkflowBuilder
    from molexp.workflow.task import Task

    class Inert(Task):
        async def execute(self, ctx):  # type: ignore[no-untyped-def, override]
            return None

    if not default_registry.has("test.inert_yaml_rt"):
        default_registry.register("test.inert_yaml_rt", lambda _config: Inert())

    spec = (
        WorkflowBuilder(name="rt")
        .add(Inert(), name="A", task_type="test.inert_yaml_rt")
        .add(Inert(), name="B", task_type="test.inert_yaml_rt", depends_on=["A"])
        .build()
    )
    text = default_compiler.spec_to_yaml(spec)
    spec2 = default_compiler.yaml_to_spec(text)
    assert default_compiler.spec_to_ir(spec) == default_compiler.spec_to_ir(spec2)
