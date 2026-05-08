"""Tests for :class:`molexp.workflow.WorkflowCompiler`.

The compiler converts between four equivalent surfaces:

- IR (dict) — the wire format
- Python script — runnable surface, IR-as-literal
- Spec — :class:`WorkflowSpec`, in-memory execution object
- Mermaid — read-only diagram

The IR↔Python round-trip is the load-bearing one (both directions must
yield bytewise-equal IR). Spec↔IR round-trip is checked at the
slugged-task level (decorator tasks are not serializable). Mermaid is a
one-way surface; we just verify the rendering covers nodes + edges.
"""

from __future__ import annotations

import pytest

from molexp.workflow.compiler import WorkflowCompiler, default_compiler


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


# ── IR → Python ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ir_to_python_emits_runnable_script_with_workflow_ir_literal():
    script = default_compiler.ir_to_python(_sample_ir())
    assert "WORKFLOW_IR" in script
    assert "WorkflowSpec.from_dict(WORKFLOW_IR)" in script
    # Single trailing newline so editors don't flag a missing final newline.
    assert script.endswith("\n")


@pytest.mark.unit
def test_ir_to_python_rejects_non_dict_input():
    with pytest.raises(ValueError, match="must be a dict"):
        default_compiler.ir_to_python("not a dict")  # type: ignore[arg-type]


@pytest.mark.unit
def test_ir_to_python_rejects_non_literal_values():
    """Callables aren't ast.literal_eval-safe; round-trip would break."""
    bad_ir = {"task_configs": [{"task_id": "t", "config": {"fn": lambda: 1}}]}
    with pytest.raises(ValueError, match="literal-safe"):
        default_compiler.ir_to_python(bad_ir)


@pytest.mark.unit
def test_ir_to_python_indents_nested_structures():
    """Empty containers render compactly; non-empty ones span multiple lines."""
    script = default_compiler.ir_to_python(
        {"name": "x", "task_configs": [], "links": [], "metadata": {}}
    )
    assert "[]" in script
    assert "{}" in script


# ── Python → IR ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_python_to_ir_round_trip_preserves_ir():
    ir = _sample_ir()
    script = default_compiler.ir_to_python(ir)
    assert default_compiler.python_to_ir(script) == ir


@pytest.mark.unit
def test_python_to_ir_rejects_missing_workflow_ir():
    with pytest.raises(ValueError, match="WORKFLOW_IR"):
        default_compiler.python_to_ir("# nothing here\n")


@pytest.mark.unit
def test_python_to_ir_rejects_invalid_python():
    with pytest.raises(ValueError, match="invalid Python"):
        default_compiler.python_to_ir("WORKFLOW_IR = {")


@pytest.mark.unit
def test_python_to_ir_rejects_non_dict_workflow_ir():
    with pytest.raises(ValueError, match="must be a dict"):
        default_compiler.python_to_ir("WORKFLOW_IR = [1, 2, 3]\n")


@pytest.mark.unit
def test_python_to_ir_ignores_unrelated_top_level_code():
    script = (
        "import os\n"
        "DEBUG = True\n"
        "WORKFLOW_IR = {'name': 'x', 'task_configs': [], 'links': []}\n"
        "print('hi')\n"
    )
    parsed = default_compiler.python_to_ir(script)
    assert parsed["name"] == "x"


# ── Mermaid ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ir_to_mermaid_emits_flowchart_header():
    out = default_compiler.ir_to_mermaid(_sample_ir())
    assert out.startswith("flowchart LR\n")
    assert out.endswith("\n")


@pytest.mark.unit
def test_ir_to_mermaid_renders_one_node_per_task():
    out = default_compiler.ir_to_mermaid(_sample_ir())
    # Both task_id labels appear; both task_types appear in the labels.
    assert "inspect" in out
    assert "train" in out
    assert "inspect_dataset" in out
    assert "train_gnn" in out


@pytest.mark.unit
def test_ir_to_mermaid_renders_one_edge_per_link():
    out = default_compiler.ir_to_mermaid(_sample_ir())
    # Mermaid IDs are namespaced with n_ to avoid digit-prefix issues.
    assert "n_inspect --> n_train" in out


@pytest.mark.unit
def test_ir_to_mermaid_handles_orphan_tasks():
    """A task with no incoming/outgoing edge still appears as a node."""
    ir = {
        "task_configs": [{"task_id": "lonely", "task_type": "noop", "config": {}}],
        "links": [],
    }
    out = default_compiler.ir_to_mermaid(ir)
    assert "n_lonely" in out
    # No '-->' anywhere — there are no edges.
    assert "-->" not in out


@pytest.mark.unit
def test_ir_to_mermaid_rejects_non_dict_input():
    with pytest.raises(ValueError, match="must be a dict"):
        default_compiler.ir_to_mermaid("oops")  # type: ignore[arg-type]


@pytest.mark.unit
def test_ir_to_mermaid_sanitizes_unsafe_ids():
    """Task IDs containing dashes / dots become underscored Mermaid IDs."""
    ir = {
        "task_configs": [{"task_id": "step-one.v2", "task_type": "x", "config": {}}],
        "links": [],
    }
    out = default_compiler.ir_to_mermaid(ir)
    assert "n_step_one_v2" in out
    # The display label keeps the original ID (inside a label literal).
    assert "step-one.v2" in out


# ── Spec ↔ IR (delegation to existing WorkflowSpec methods) ──────────────────


@pytest.mark.unit
def test_ir_to_spec_delegates_to_workflow_spec():
    """Smoke test that the compiler routes through ``WorkflowSpec.from_dict``."""
    from molexp.workflow.registry import default_registry

    # Register a no-op task type so from_dict can resolve the slug.
    # The registry expects either a factory callable taking ``config`` or
    # a class whose ``__init__`` accepts ``**config``.
    class _Noop:
        async def execute(self, ctx):
            return None

    default_registry.register("noop_for_compiler_test", lambda cfg: _Noop())
    try:
        ir = {
            "name": "demo",
            "task_configs": [
                {"task_id": "t1", "task_type": "noop_for_compiler_test", "config": {}}
            ],
            "links": [],
        }
        spec = default_compiler.ir_to_spec(ir)
        assert spec.name == "demo"
        # Round-trip back to IR keeps the slug.
        round_tripped = default_compiler.spec_to_ir(spec)
        assert round_tripped["task_configs"][0]["task_type"] == "noop_for_compiler_test"
        assert round_tripped["task_configs"][0]["task_id"] == "t1"
    finally:
        # Cleanup so the registry stays scoped to this test.
        default_registry._factories.pop("noop_for_compiler_test", None)  # type: ignore[attr-defined]


# ── Compiler instance hygiene ────────────────────────────────────────────────


@pytest.mark.unit
def test_compiler_subclass_can_override_one_method():
    """A subclass should be able to swap a single converter."""

    class CustomCompiler(WorkflowCompiler):
        def ir_to_mermaid(self, ir: dict) -> str:  # type: ignore[override]
            return "custom-mermaid\n"

    c = CustomCompiler()
    assert c.ir_to_mermaid(_sample_ir()) == "custom-mermaid\n"
    # Other converters still work.
    ir = _sample_ir()
    assert c.python_to_ir(c.ir_to_python(ir)) == ir
