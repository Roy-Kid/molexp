"""``WorkflowBindingRegistry`` — replaces ``workflow/bindings.py``.

Binding moved off the workflow artifact onto a dedicated registry.
The five old module-level binding functions and the legacy
``WorkflowSpec`` / ``Workflow`` names are gone; binding now flows
through ``WorkflowBindingRegistry`` (and the process-global
``default_binding_registry``).
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    CompiledWorkflow,
    WorkflowBindingRegistry,
    WorkflowCompiler,
    default_binding_registry,
)


class _StubExperiment:
    def __init__(self, exp_id: str) -> None:
        self.id = exp_id


@pytest.fixture(autouse=True)
def _isolate_registry():
    default_binding_registry.clear()
    yield
    default_binding_registry.clear()


def _make_spec(name: str = "wf") -> CompiledWorkflow:
    return WorkflowCompiler(name=name).compile()


# ── Rename + module-level deletions ────────────────────────────────────────


def test_compiled_workflow_class_name():
    assert CompiledWorkflow.__name__ == "CompiledWorkflow"


def test_workflow_builder_class_name():
    assert WorkflowCompiler.__name__ == "WorkflowCompiler"


def test_old_workflowspec_export_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import WorkflowSpec  # noqa: F401


def test_old_workflow_export_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import Workflow  # noqa: F401


def test_set_workflow_module_function_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import set_workflow  # noqa: F401


def test_get_workflow_module_function_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import get_workflow  # noqa: F401


def test_has_workflow_module_function_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import has_workflow  # noqa: F401


def test_clear_workflow_module_function_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import clear_workflow  # noqa: F401


def test_reset_bindings_module_function_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import reset_bindings  # noqa: F401


def test_bindings_module_is_gone():
    with pytest.raises(ImportError):
        import molexp.workflow.bindings  # noqa: F401


# ── bind / for_experiment / is_bound / unbind ──────────────────────────────


def test_bind_then_for_experiment_returns_same_spec():
    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    reg.bind(exp, spec)
    assert reg.for_experiment(exp) is spec


def test_for_experiment_returns_none_when_unbound():
    reg = WorkflowBindingRegistry()
    exp = _StubExperiment("never-bound")
    assert reg.for_experiment(exp) is None


def test_is_bound_reflects_binding_state():
    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    assert reg.is_bound(exp, spec) is False
    reg.bind(exp, spec)
    assert reg.is_bound(exp, spec) is True


def test_unbind_returns_true_when_present_false_when_absent():
    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    assert reg.unbind(exp) is False
    reg.bind(exp, spec)
    assert reg.unbind(exp) is True
    assert reg.is_bound(exp, spec) is False


def test_rebinding_overwrites_previous_spec():
    reg = WorkflowBindingRegistry()
    s1 = _make_spec("a")
    s2 = _make_spec("b")
    exp = _StubExperiment("e1")
    reg.bind(exp, s1)
    reg.bind(exp, s2)
    assert reg.for_experiment(exp) is s2


def test_clear_clears_all_bindings():
    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    reg.bind(exp, spec)
    reg.clear()
    assert reg.for_experiment(exp) is None


def test_default_binding_registry_is_a_registry():
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    default_binding_registry.bind(exp, spec)
    assert default_binding_registry.for_experiment(exp) is spec


# ── Validation: ValueError preserved from old free function ────────────────


def test_bind_rejects_non_workflow_target():
    reg = WorkflowBindingRegistry()
    not_an_exp = object()
    spec = _make_spec("a")
    with pytest.raises(ValueError):
        reg.bind(not_an_exp, spec)


def test_bind_rejects_experiment_with_non_string_id():
    class _BadExp:
        id = 42

    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    with pytest.raises(ValueError):
        reg.bind(_BadExp(), spec)


def test_bind_rejects_experiment_with_empty_id():
    class _EmptyIdExp:
        id = ""

    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    with pytest.raises(ValueError):
        reg.bind(_EmptyIdExp(), spec)
