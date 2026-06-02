"""Workflow ↔ experiment binding — explicit registry, no process-global.

Replaces the old ``Workflow._bindings_registry`` ``ClassVar`` (an
un-injectable process-global that leaked across tests via a private
``_reset_registry`` hook). A :class:`WorkflowBindingRegistry` is an
explicit, injectable ``{experiment_id → CompiledWorkflow}`` store passed
into :meth:`WorkflowCompiler.compile` via ``registry=``; a module-level
:data:`default_binding_registry` provides the per-process default (mirroring
``default_codec`` / ``default_registry``), and ``registry.clear()`` is the
public test-isolation hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .compiled import CompiledWorkflow


class _ExperimentLike(Protocol):
    """Duck-typed handle to anything with a stable string ``id``."""

    @property
    def id(self) -> str: ...


def _exp_id(experiment: _ExperimentLike) -> str | None:
    exp_id = getattr(experiment, "id", None)
    return exp_id if isinstance(exp_id, str) and exp_id else None


class WorkflowBinding(BaseModel):
    """Immutable record that a workflow is bound to an experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    workflow_id: str


class WorkflowBindingRegistry:
    """Explicit ``{experiment_id → CompiledWorkflow}`` store.

    Injectable (pass into ``compile(registry=…)``); the default instance is
    :data:`default_binding_registry`. Unlike the old class-attribute global,
    a test can construct a fresh registry instead of mutating shared state,
    and :meth:`clear` is a normal public method (not a private reset hook).
    """

    def __init__(self) -> None:
        self._by_experiment: dict[str, CompiledWorkflow] = {}

    def bind(self, experiment: _ExperimentLike, compiled: CompiledWorkflow) -> WorkflowBinding:
        """Bind *compiled* to *experiment*; return the :class:`WorkflowBinding`."""
        exp_id = _exp_id(experiment)
        if exp_id is None:
            raise ValueError(
                f"bind expects an experiment with a non-empty string `id`; got {experiment!r}"
            )
        self._by_experiment[exp_id] = compiled
        return WorkflowBinding(experiment_id=exp_id, workflow_id=compiled.workflow_id)

    def for_experiment(self, experiment: _ExperimentLike) -> CompiledWorkflow | None:
        """Return the compiled workflow bound to *experiment*, or ``None``."""
        exp_id = _exp_id(experiment)
        if exp_id is None:
            return None
        return self._by_experiment.get(exp_id)

    def unbind(self, experiment: _ExperimentLike) -> bool:
        """Drop the binding for *experiment*. Returns True iff one existed."""
        exp_id = _exp_id(experiment)
        if exp_id is None:
            return False
        return self._by_experiment.pop(exp_id, None) is not None

    def is_bound(self, experiment: _ExperimentLike, compiled: CompiledWorkflow) -> bool:
        """Return True iff *compiled* is the artifact bound to *experiment*."""
        exp_id = _exp_id(experiment)
        if exp_id is None:
            return False
        return self._by_experiment.get(exp_id) is compiled

    def clear(self) -> None:
        """Drop every binding. Public test-isolation hook."""
        self._by_experiment.clear()


default_binding_registry = WorkflowBindingRegistry()


__all__ = [
    "WorkflowBinding",
    "WorkflowBindingRegistry",
    "default_binding_registry",
]
