"""``Workflow``-class binding registry — replaces ``workflow/bindings.py``.

Covers acceptance criteria ac-004, ac-005 of the
``oop-api-rectification`` spec: rename ``WorkflowSpec`` → ``Workflow``
(and old ``Workflow`` → ``WorkflowBuilder``), and absorb the five
module-level binding functions onto the renamed class as instance
methods + classmethod + private test-isolation hook.
"""

from __future__ import annotations

import pytest

from molexp.workflow import Workflow, WorkflowBuilder


class _StubExperiment:
    def __init__(self, exp_id: str) -> None:
        self.id = exp_id


@pytest.fixture(autouse=True)
def _isolate_registry():
    Workflow._reset_registry()
    yield
    Workflow._reset_registry()


def _make_spec(name: str = "wf") -> Workflow:
    return WorkflowBuilder(name=name).build()


# ── Rename + module-level deletions ────────────────────────────────────────


def test_workflow_class_name():
    assert Workflow.__name__ == "Workflow"


def test_workflow_builder_class_name():
    assert WorkflowBuilder.__name__ == "WorkflowBuilder"


def test_old_workflowspec_export_is_gone():
    with pytest.raises(ImportError):
        from molexp.workflow import WorkflowSpec  # noqa: F401


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


# ── bind_to / for_experiment / is_bound_to / unbind_from ───────────────────


def test_bind_to_then_for_experiment_returns_same_spec():
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    spec.bind_to(exp)
    assert Workflow.for_experiment(exp) is spec


def test_for_experiment_returns_none_when_unbound():
    exp = _StubExperiment("never-bound")
    assert Workflow.for_experiment(exp) is None


def test_is_bound_to_reflects_binding_state():
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    assert spec.is_bound_to(exp) is False
    spec.bind_to(exp)
    assert spec.is_bound_to(exp) is True


def test_unbind_from_returns_true_when_present_false_when_absent():
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    assert spec.unbind_from(exp) is False
    spec.bind_to(exp)
    assert spec.unbind_from(exp) is True
    assert spec.is_bound_to(exp) is False


def test_rebinding_overwrites_previous_spec():
    s1 = _make_spec("a")
    s2 = _make_spec("b")
    exp = _StubExperiment("e1")
    s1.bind_to(exp)
    s2.bind_to(exp)
    assert Workflow.for_experiment(exp) is s2


def test_reset_registry_clears_all_bindings():
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    spec.bind_to(exp)
    Workflow._reset_registry()
    assert Workflow.for_experiment(exp) is None


# ── Validation: TypeError / ValueError preserved from old free function ────


def test_bind_to_rejects_non_workflow_target():
    not_an_exp = object()
    spec = _make_spec("a")
    # bind_to is on Workflow; calling it with bogus argument should fail
    # the experiment-shape check
    with pytest.raises(ValueError):
        spec.bind_to(not_an_exp)


def test_bind_to_rejects_experiment_with_non_string_id():
    class _BadExp:
        id = 42

    spec = _make_spec("a")
    with pytest.raises(ValueError):
        spec.bind_to(_BadExp())


def test_bind_to_rejects_experiment_with_empty_id():
    class _EmptyIdExp:
        id = ""

    spec = _make_spec("a")
    with pytest.raises(ValueError):
        spec.bind_to(_EmptyIdExp())
