"""Tests for :class:`molexp.workflow.codec.WorkflowCodec`.

The codec converts between four equivalent surfaces: IR (dict, the wire
format), a runnable Python script (IR-as-literal), a :class:`CompiledWorkflow`
spec, and read-only Mermaid. The IR↔Python round-trip is the load-bearing one;
spec↔IR round-tripping is owned by ``test_ir_roundtrip``. This file pins the
codec's rendering surfaces, the "never exec user code" extraction, and the
single-owner delegation from ``CompiledWorkflow.to_ir``. The strict typed-edge
``link.json`` schema (kind required) is pinned at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from molexp.workflow.codec import default_codec

_GOLDEN = Path(__file__).parent / "golden"


def _sample_ir() -> dict:
    return {
        "name": "qm9-gnn-baseline",
        "task_configs": [
            {"task_id": "inspect", "task_type": "inspect_dataset", "config": {"path": "qm9.h5"}},
            {"task_id": "train", "task_type": "train_gnn", "config": {"epochs": 50, "lr": 1e-3}},
        ],
        "links": [{"source": "inspect", "target": "train"}],
        "metadata": {},
    }


def _register_golden_task_types() -> None:
    from molexp.workflow.registry import default_registry

    class _Noop:
        async def execute(self, ctx):
            return None

    def _factory(cfg):
        inst = _Noop()
        inst._task_config = dict(cfg)
        return inst

    for slug in ("golden_inspect", "golden_train"):
        if slug not in default_registry._factories:  # type: ignore[attr-defined]
            default_registry.register(slug, _factory)


class TestWorkflowCodec:
    """The codec's four surfaces + delegation contract."""

    @pytest.mark.unit
    def test_ir_to_python_matches_golden(self) -> None:
        """IR → runnable script renders byte-identically (formatting is a feature —
        the script is diff-reviewable)."""
        ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
        expected = (_GOLDEN / "sample.py.txt").read_text()
        assert default_codec.ir_to_python(ir) == expected

    @pytest.mark.unit
    def test_python_to_ir_round_trips(self) -> None:
        """``python_to_ir(ir_to_python(ir)) == ir`` — the load-bearing inverse."""
        ir = _sample_ir()
        assert default_codec.python_to_ir(default_codec.ir_to_python(ir)) == ir

    @pytest.mark.unit
    def test_ir_to_python_rejects_non_literal_values(self) -> None:
        """Callables aren't ``ast.literal_eval``-safe; the round-trip would break."""
        bad_ir = {"task_configs": [{"task_id": "t", "config": {"fn": lambda: 1}}]}
        with pytest.raises(ValueError, match="literal-safe"):
            default_codec.ir_to_python(bad_ir)

    @pytest.mark.unit
    def test_python_to_ir_rejects_missing_workflow_ir(self) -> None:
        with pytest.raises(ValueError, match="WORKFLOW_IR"):
            default_codec.python_to_ir("# nothing here\n")

    @pytest.mark.unit
    def test_python_to_ir_ignores_non_workflow_ir_code(self) -> None:
        """Only the ``WORKFLOW_IR`` literal is extracted — imports, other
        assignments, and calls are ignored (user code is never executed)."""
        script = (
            "import os\n"
            "DEBUG = True\n"
            "WORKFLOW_IR = {'name': 'x', 'task_configs': [], 'links': []}\n"
            "print('hi')\n"
        )
        assert default_codec.python_to_ir(script)["name"] == "x"

    @pytest.mark.unit
    def test_ir_to_mermaid_matches_golden(self) -> None:
        """IR → Mermaid is one-way; the golden covers node + edge rendering."""
        ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
        expected = (_GOLDEN / "sample.mermaid.txt").read_text()
        assert default_codec.ir_to_mermaid(ir) == expected

    @pytest.mark.unit
    def test_ir_to_mermaid_sanitizes_unsafe_ids(self) -> None:
        """Task IDs containing dashes / dots become underscored Mermaid IDs."""
        ir = {
            "task_configs": [{"task_id": "step-one.v2", "task_type": "x", "config": {}}],
            "links": [],
        }
        out = default_codec.ir_to_mermaid(ir)
        assert "n_step_one_v2" in out
        assert "step-one.v2" in out

    @pytest.mark.unit
    def test_compiled_to_ir_delegates_to_codec(self) -> None:
        """``CompiledWorkflow.to_ir`` is a thin delegator — the codec is the
        single owner of IR conversion (spec ``workflow-refactor-01`` ac-004)."""
        _register_golden_task_types()
        ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
        spec = default_codec.ir_to_spec(ir)
        assert spec.to_ir() == default_codec.spec_to_ir(spec)


class TestLinkSchema:
    """Strict typed-edge ``link.json`` schema (flowgram-workflow-canvas-01)."""

    _SCHEMA_DIR = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "schema"

    @pytest.mark.unit
    def test_link_requires_a_valid_kind(self) -> None:
        """``kind`` is required with a fixed enum: a link missing ``kind`` fails
        (no default-to-data at the schema layer), and an out-of-enum kind fails."""
        link_schema = json.loads((self._SCHEMA_DIR / "link.json").read_text())
        validator = jsonschema.Draft7Validator(link_schema)

        valid = {
            "source": "Inspect_aa11bb22",
            "target": "Train_cc33dd44",
            "mapping": {},
            "status": "pending",
            "kind": "data",
        }
        validator.validate(valid)  # does not raise

        with pytest.raises(jsonschema.ValidationError):
            validator.validate({k: v for k, v in valid.items() if k != "kind"})
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({**valid, "kind": "bogus"})
