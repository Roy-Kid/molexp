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

from molexp.workflow.codec import WorkflowCodec, default_codec

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


@pytest.mark.unit
def test_workflow_codec_is_public_surface():
    """`WorkflowCodec` + `default_codec` import from the package root (ac-002)."""
    from molexp.workflow import WorkflowCodec as PublicCodec
    from molexp.workflow import default_codec as public_default

    assert PublicCodec is WorkflowCodec
    assert isinstance(public_default, WorkflowCodec)
    assert isinstance(default_codec, WorkflowCodec)


# ── ac-003: byte-identical output to captured pre-refactor golden ────────────


def _register_golden_task_types():
    from molexp.workflow.registry import default_registry

    class _Noop:
        async def execute(self, ctx):
            return None

    for slug in ("golden_inspect", "golden_train"):
        if slug not in default_registry._factories:  # type: ignore[attr-defined]
            default_registry.register(slug, lambda cfg: _Noop())  # noqa: ARG005


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


@pytest.mark.unit
def test_compiled_from_ir_delegates_to_default_codec():
    """`CompiledWorkflow.from_ir(ir)` and `default_codec.ir_to_spec(ir)` agree."""
    from molexp.workflow import CompiledWorkflow

    _register_golden_task_types()
    ir = json.loads((_GOLDEN / "sample_ir.json").read_text())
    via_compiled = CompiledWorkflow.from_ir(ir).to_ir()
    via_codec = default_codec.spec_to_ir(default_codec.ir_to_spec(ir))
    assert via_compiled == via_codec


@pytest.mark.unit
def test_execution_route_uses_default_codec():
    """`server/routes/execution.py` routes IR through the codec, not Workflow.from_dict (ac-004)."""
    route = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "molexp"
        / "server"
        / "routes"
        / "execution.py"
    )
    text = route.read_text()
    assert "default_codec.ir_to_spec(" in text
    assert "Workflow.from_dict(" not in text


# ── IR → Python ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ir_to_python_emits_runnable_script_with_workflow_ir_literal():
    script = default_codec.ir_to_python(_sample_ir())
    assert "WORKFLOW_IR" in script
    assert "CompiledWorkflow.from_ir(WORKFLOW_IR)" in script
    assert script.endswith("\n")


@pytest.mark.unit
def test_ir_to_python_rejects_non_dict_input():
    with pytest.raises(ValueError, match="must be a dict"):
        default_codec.ir_to_python("not a dict")  # type: ignore[arg-type]


@pytest.mark.unit
def test_ir_to_python_rejects_non_literal_values():
    """Callables aren't ast.literal_eval-safe; round-trip would break."""
    bad_ir = {"task_configs": [{"task_id": "t", "config": {"fn": lambda: 1}}]}
    with pytest.raises(ValueError, match="literal-safe"):
        default_codec.ir_to_python(bad_ir)


@pytest.mark.unit
def test_ir_to_python_indents_nested_structures():
    """Empty containers render compactly; non-empty ones span multiple lines."""
    script = default_codec.ir_to_python(
        {"name": "x", "task_configs": [], "links": [], "metadata": {}}
    )
    assert "[]" in script
    assert "{}" in script


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
def test_python_to_ir_rejects_invalid_python():
    with pytest.raises(ValueError, match="invalid Python"):
        default_codec.python_to_ir("WORKFLOW_IR = {")


@pytest.mark.unit
def test_python_to_ir_rejects_non_dict_workflow_ir():
    with pytest.raises(ValueError, match="must be a dict"):
        default_codec.python_to_ir("WORKFLOW_IR = [1, 2, 3]\n")


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
def test_ir_to_mermaid_emits_flowchart_header():
    out = default_codec.ir_to_mermaid(_sample_ir())
    assert out.startswith("flowchart LR\n")
    assert out.endswith("\n")


@pytest.mark.unit
def test_ir_to_mermaid_renders_one_node_per_task():
    out = default_codec.ir_to_mermaid(_sample_ir())
    assert "inspect" in out
    assert "train" in out
    assert "inspect_dataset" in out
    assert "train_gnn" in out


@pytest.mark.unit
def test_ir_to_mermaid_renders_one_edge_per_link():
    out = default_codec.ir_to_mermaid(_sample_ir())
    assert "n_inspect --> n_train" in out


@pytest.mark.unit
def test_ir_to_mermaid_handles_orphan_tasks():
    """A task with no incoming/outgoing edge still appears as a node."""
    ir = {
        "task_configs": [{"task_id": "lonely", "task_type": "noop", "config": {}}],
        "links": [],
    }
    out = default_codec.ir_to_mermaid(ir)
    assert "n_lonely" in out
    assert "-->" not in out


@pytest.mark.unit
def test_ir_to_mermaid_rejects_non_dict_input():
    with pytest.raises(ValueError, match="must be a dict"):
        default_codec.ir_to_mermaid("oops")  # type: ignore[arg-type]


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


@pytest.mark.unit
def test_ir_to_spec_builds_workflow():
    """The codec owns IR → Workflow construction."""
    from molexp.workflow.registry import default_registry

    class _Noop:
        async def execute(self, ctx):
            return None

    default_registry.register("noop_for_codec_test", lambda cfg: _Noop())  # noqa: ARG005
    try:
        ir = {
            "name": "demo",
            "task_configs": [{"task_id": "t1", "task_type": "noop_for_codec_test", "config": {}}],
            "links": [],
        }
        spec = default_codec.ir_to_spec(ir)
        assert spec.name == "demo"
        round_tripped = default_codec.spec_to_ir(spec)
        assert round_tripped["task_configs"][0]["task_type"] == "noop_for_codec_test"
        assert round_tripped["task_configs"][0]["task_id"] == "t1"
    finally:
        default_registry._factories.pop("noop_for_codec_test", None)  # type: ignore[attr-defined]


# ── Codec instance hygiene ────────────────────────────────────────────────────


@pytest.mark.unit
def test_codec_subclass_can_override_one_method():
    """A subclass should be able to swap a single converter."""

    class CustomCodec(WorkflowCodec):
        def ir_to_mermaid(self, ir: dict) -> str:  # type: ignore[override]
            return "custom-mermaid\n"

    c = CustomCodec()
    assert c.ir_to_mermaid(_sample_ir()) == "custom-mermaid\n"
    ir = _sample_ir()
    assert c.python_to_ir(c.ir_to_python(ir)) == ir
