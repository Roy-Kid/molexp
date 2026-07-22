"""Round-trip + safety tests for the YAML / contract surface of
:class:`molexp.workflow.codec.WorkflowCodec`.

The plain IR↔Python↔spec surfaces live in ``test_codec.py``; this file owns the
YAML surface and the contract sidecar it carries:

- ``WorkflowContract`` → dict → YAML → dict → contract is field-equal.
- ``spec_to_yaml`` ⇄ ``yaml_to_spec`` is byte-stable through the JSON IR.
- An old IR JSON without ``workflow_contract`` survives the spec round-trip with
  no contract injected (back-compat).
- ``yaml_to_ir`` refuses unsafe YAML tags (``safe_load``) and non-dict roots.
"""

from __future__ import annotations

import pytest
import yaml

from molexp.workflow.codec import default_codec
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


class TestWorkflowCodecYamlRoundTrip:
    def test_contract_survives_dict_yaml_dict_chain_field_equal(self) -> None:
        contract = _sample_contract()
        text = default_codec.ir_to_yaml(default_codec.contract_to_dict(contract))
        rebuilt = default_codec.dict_to_contract(default_codec.yaml_to_ir(text))
        assert rebuilt == contract

    def test_spec_survives_yaml_round_trip_through_ir(self) -> None:
        """``spec_to_yaml`` ⇄ ``yaml_to_spec`` is IR-stable (slugged tasks only)."""
        from molexp.workflow.compiler import WorkflowCompiler
        from molexp.workflow.registry import default_registry
        from molexp.workflow.task import Task

        class Inert(Task):
            async def execute(self, ctx):  # type: ignore[no-untyped-def, override]
                return None

        if not default_registry.has("test.inert_yaml_rt"):
            default_registry.register("test.inert_yaml_rt", Inert)

        spec = (
            WorkflowCompiler(name="rt")
            .add(Inert(), name="A")
            .add(Inert(), name="B", depends_on=["A"])
            .compile()
        )
        text = default_codec.spec_to_yaml(spec)
        spec2 = default_codec.yaml_to_spec(text)
        assert default_codec.spec_to_ir(spec) == default_codec.spec_to_ir(spec2)

    def test_old_ir_without_contract_stays_contract_free(self) -> None:
        """An IR JSON with no ``workflow_contract`` key must not gain one
        across ``ir_to_spec`` → ``spec_to_ir``."""
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
        spec = default_codec.ir_to_spec(ir_in)
        ir_out = default_codec.spec_to_ir(spec)
        assert "workflow_contract" not in ir_out


class TestWorkflowCodecYamlSafety:
    def test_yaml_to_ir_rejects_python_object_tag(self) -> None:
        """``!!python/object`` is rejected by ``safe_load`` — the codec forwards
        the error rather than constructing an arbitrary object."""
        unsafe = "!!python/object/apply:os.system [echo hello]\n"
        with pytest.raises(yaml.YAMLError):
            default_codec.yaml_to_ir(unsafe)

    def test_yaml_to_ir_rejects_non_dict_root(self) -> None:
        """A list-rooted YAML doc isn't an IR shape; the loader rejects it."""
        with pytest.raises(ValueError):
            default_codec.yaml_to_ir("- a\n- b\n")
