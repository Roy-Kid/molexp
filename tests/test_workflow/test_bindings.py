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


# ── bind / for_experiment / is_bound / unbind ──────────────────────────────


def test_bind_then_for_experiment_returns_same_spec():
    reg = WorkflowBindingRegistry()
    spec = _make_spec("a")
    exp = _StubExperiment("e1")
    reg.bind(exp, spec)
    assert reg.for_experiment(exp) is spec


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


# ── Validation: ValueError preserved from old free function ────────────────


def test_bind_rejects_non_workflow_target():
    reg = WorkflowBindingRegistry()
    not_an_exp = object()
    spec = _make_spec("a")
    with pytest.raises(ValueError):
        reg.bind(not_an_exp, spec)
