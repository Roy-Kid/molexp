"""Process-local registry pairing an :class:`Experiment` with a :class:`WorkflowSpec`.

Workspace does not store the experiment-to-workflow pairing
(rectification 2026-05-09 — workspace is the storage primitive,
workflow is the engine, the pairing is the caller's concern). But
real applications — the CLI, the FastAPI server, the agent's native
``set_workflow_from_ir`` tool — still need to *remember* which
workflow goes with which experiment within a single process.

This module is the workflow layer's answer: a process-local dict
keyed by ``experiment.id``. Bindings persist for the lifetime of the
Python process and survive across function boundaries, but they do
not survive a process restart on their own. For cross-process
durability — the cluster-worker scenario where a worker re-imports
the user script — pair this registry with
:func:`molexp.workflow.resolve_spec_entrypoint` so the worker can
recover the binding by re-running the user's
``set_workflow(experiment, spec)`` call.

Usage::

    from molexp.workflow import set_workflow, get_workflow

    exp = project.experiment("baseline", params={"lr": 1e-3})
    set_workflow(exp, my_workflow_spec)

    # later, possibly in another function:
    spec = get_workflow(exp)
    if spec is None:
        raise RuntimeError(...)
    await spec.execute(run=run)

Why a module-level dict and not a class? The registry is implicitly
global within a process. A class would invite spurious instance
juggling (whose registry is the canonical one?). The function-level
API matches the conceptual scope: "the workflow this experiment is
bound to in *this* process".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .spec import WorkflowSpec

if TYPE_CHECKING:
    pass


class _ExperimentLike(Protocol):
    """Duck-typed handle to anything with a stable string ``id``.

    Workspace's :class:`Experiment` satisfies this; tests that
    construct a stand-in object with an ``id`` attribute also work.
    The workflow layer never imports ``molexp.workspace.Experiment``
    directly — keeping the dependency direction one-way.
    """

    @property
    def id(self) -> str: ...


_bindings: dict[str, WorkflowSpec] = {}


def set_workflow(experiment: _ExperimentLike, spec: WorkflowSpec) -> None:
    """Bind *spec* to *experiment* in the current process.

    Re-binding the same experiment overwrites the previous spec —
    the caller controls overwrite semantics. (The legacy
    ``Experiment.set_workflow`` raised on double-bind; that
    enforcement is intentionally not preserved because the typical
    cross-process worker flow re-runs the user script and needs to
    re-bind cleanly.)

    Args:
        experiment: Anything with a stable string ``id``. In
            production this is :class:`molexp.workspace.Experiment`.
        spec: The compiled workflow spec to associate with the
            experiment.

    Raises:
        TypeError: If *spec* is not a :class:`WorkflowSpec`.
        ValueError: If *experiment* has no string ``id``.
    """
    if not isinstance(spec, WorkflowSpec):
        raise TypeError(f"set_workflow expects a WorkflowSpec; got {type(spec).__name__}")
    exp_id = getattr(experiment, "id", None)
    if not isinstance(exp_id, str) or not exp_id:
        raise ValueError(
            f"set_workflow expects an experiment with a non-empty string `id`; got {experiment!r}"
        )
    _bindings[exp_id] = spec


def get_workflow(experiment: _ExperimentLike) -> WorkflowSpec | None:
    """Return the spec bound to *experiment*, or ``None`` if unbound."""
    exp_id = getattr(experiment, "id", None)
    if not isinstance(exp_id, str) or not exp_id:
        return None
    return _bindings.get(exp_id)


def has_workflow(experiment: _ExperimentLike) -> bool:
    """``True`` iff a spec is currently bound to *experiment*."""
    return get_workflow(experiment) is not None


def clear_workflow(experiment: _ExperimentLike) -> bool:
    """Drop the binding for *experiment*. Returns ``True`` iff one existed."""
    exp_id = getattr(experiment, "id", None)
    if not isinstance(exp_id, str) or not exp_id:
        return False
    return _bindings.pop(exp_id, None) is not None


def reset_bindings() -> None:
    """Clear every binding in the registry. Mainly for test isolation."""
    _bindings.clear()


__all__ = [
    "clear_workflow",
    "get_workflow",
    "has_workflow",
    "reset_bindings",
    "set_workflow",
]
