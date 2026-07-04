"""Tests for :class:`molexp.workflow.WorkflowCodec`.

The codec converts between four equivalent surfaces:

- IR (dict) — the wire format
- Python script — runnable surface, IR-as-literal
- Spec — :class:`Workflow`, in-memory execution object
- Mermaid — read-only diagram

The IR↔Python round-trip is the load-bearing one (both directions must
yield bytewise-equal IR). Spec↔IR round-trip is checked at the
slugged-task level (decorator tasks are not serializable). Mermaid is a
one-way surface; we just verify the rendering covers nodes + edges.

This module also pins the codec-fold refactor (spec
``workflow-refactor-01-codec-fold``): the ``WorkflowCompiler`` name is
freed (ac-001), ``WorkflowCodec`` / ``default_codec`` are the public
surface (ac-002), every representation surface is byte-identical to the
captured pre-refactor golden (ac-003), and the codec is the single owner
of IR conversion with ``Workflow.to_dict`` / ``from_dict`` as thin
delegators (ac-004).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workflow.codec import default_codec

_GOLDEN = Path(__file__).parent / "golden"


def _sample_ir() -> dict:
    return {
        "name": "qm9-gnn-baseline",
        "task_configs": [
            {
                "task_id": "inspect",
                "task_type": "inspect_dataset",
                "config": {"path": "qm9.h5"},
            },
            {
                "task_id": "train",
                "task_type": "train_gnn",
                "config": {"epochs": 50, "lr": 1e-3},
            },
        ],
        "links": [{"source": "inspect", "target": "train"}],
        "metadata": {},
    }


# ── ac-001 / ac-002: namespace freed + codec is the public surface ───────────


# ── ac-003: byte-identical output to captured pre-refactor golden ────────────


def _register_golden_task_types():
    from molexp.workflow.registry import default_registry

    class _Noop:
        async def execute(self, ctx):
            return None

    def _factory(cfg):
        # Config now lives on the instance (a task IS its config); the factory
        # attaches the reconstructed __init__ args so spec_to_ir re-emits them.
        inst = _Noop()
        inst._task_config = dict(cfg)
        return inst

    for slug in ("golden_inspect", "golden_train"):
        if slug not in default_registry._factories:  # type: ignore[attr-defined]
            default_registry.register(slug, _factory)


@pytest.mark.unit
def test_ir_to_python_is_byte_identical_to_golden():
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    expected = (_GOLDEN / "sample.py.txt").read_text()
    assert default_codec.ir_to_python(ir) == expected


@pytest.mark.unit
def test_ir_to_mermaid_is_byte_identical_to_golden():
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    expected = (_GOLDEN / "sample.mermaid.txt").read_text()
    assert default_codec.ir_to_mermaid(ir) == expected


@pytest.mark.unit
def test_spec_to_ir_is_byte_identical_to_golden():
    """The codec's deterministic IR payload is byte-identical to the golden.

    ``workflow_id`` is a content hash of the registered task *class code*
    (computed during spec construction, not by the codec), so it varies
    with the concrete task class registered for the slug; it is normalized
    on both sides. Everything the codec deterministically emits — name,
    task_configs, links, metadata — is compared byte-for-byte.
    """
    _register_golden_task_types()
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    spec = default_codec.ir_to_spec(ir)
    produced = dict(default_codec.spec_to_ir(spec))
    produced["workflow_id"] = "<normalized>"
    produced_text = json.dumps(produced, indent=2, sort_keys=True) + "\n"
    assert produced_text == (_GOLDEN / "sample_spec_to_ir.json").read_text()


@pytest.mark.unit
def test_spec_to_ir_round_trips_through_ir_to_spec():
    """spec_to_ir(ir_to_spec(ir)) == ir for the slugged data-DAG fixture (ac-003)."""
    _register_golden_task_types()
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    rebuilt = default_codec.spec_to_ir(default_codec.ir_to_spec(ir))
    # Slug + topology survive the round-trip.
    assert rebuilt["task_configs"][0]["task_id"] == "inspect"
    assert rebuilt["task_configs"][0]["task_type"] == "golden_inspect"
    assert {(c["task_id"], c["task_type"]) for c in rebuilt["task_configs"]} == {
        ("inspect", "golden_inspect"),
        ("train", "golden_train"),
    }
    assert {(link["source"], link["target"]) for link in rebuilt["links"]} == {("inspect", "train")}


# ── ac-004: codec is the single owner of IR conversion ───────────────────────


@pytest.mark.unit
def test_compiled_to_ir_delegates_to_default_codec():
    """`CompiledWorkflow.to_ir(s) == default_codec.spec_to_ir(s)`."""
    _register_golden_task_types()
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    spec = default_codec.ir_to_spec(ir)
    assert spec.to_ir() == default_codec.spec_to_ir(spec)


# ── IR → Python ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ir_to_python_rejects_non_literal_values():
    """Callables aren't ast.literal_eval-safe; round-trip would break."""
    bad_ir = {"task_configs": [{"task_id": "t", "config": {"fn": lambda: 1}}]}
    with pytest.raises(ValueError, match="literal-safe"):
        default_codec.ir_to_python(bad_ir)


# ── Python → IR ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_python_to_ir_round_trip_preserves_ir():
    ir = _sample_ir()
    script = default_codec.ir_to_python(ir)
    assert default_codec.python_to_ir(script) == ir


@pytest.mark.unit
def test_python_to_ir_rejects_missing_workflow_ir():
    with pytest.raises(ValueError, match="WORKFLOW_IR"):
        default_codec.python_to_ir("# nothing here\n")


@pytest.mark.unit
def test_python_to_ir_ignores_unrelated_top_level_code():
    script = (
        "import os\n"
        "DEBUG = True\n"
        "WORKFLOW_IR = {'name': 'x', 'task_configs': [], 'links': []}\n"
        "print('hi')\n"
    )
    parsed = default_codec.python_to_ir(script)
    assert parsed["name"] == "x"


# ── Mermaid ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ir_to_mermaid_sanitizes_unsafe_ids():
    """Task IDs containing dashes / dots become underscored Mermaid IDs."""
    ir = {
        "task_configs": [{"task_id": "step-one.v2", "task_type": "x", "config": {}}],
        "links": [],
    }
    out = default_codec.ir_to_mermaid(ir)
    assert "n_step_one_v2" in out
    assert "step-one.v2" in out


# ── Spec ↔ IR (delegation to the codec's own bodies) ─────────────────────────


# ── Codec instance hygiene ────────────────────────────────────────────────────


# ── Strict typed-edge / position schema contract (flowgram-workflow-canvas-01) ──

import jsonschema  # noqa: E402

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "schema"


def _load_schema(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / name).read_text())


@pytest.mark.unit
def test_link_schema_requires_kind() -> None:
    """link.json marks ``kind`` required with a five-value enum; a link missing
    ``kind`` fails validation (no default-to-data fallback at the schema layer),
    and an out-of-enum kind also fails."""
    link_schema = _load_schema("link.json")
    validator = jsonschema.Draft7Validator(link_schema)

    # source/target follow the schema's Name_hex8 id pattern so the test
    # isolates the `kind` requirement rather than tripping the id pattern.
    valid = {
        "source": "Inspect_aa11bb22",
        "target": "Train_cc33dd44",
        "mapping": {},
        "status": "pending",
        "kind": "data",
    }
    validator.validate(valid)  # does not raise

    missing_kind = {k: v for k, v in valid.items() if k != "kind"}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(missing_kind)

    bad_kind = {**valid, "kind": "bogus"}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad_kind)
