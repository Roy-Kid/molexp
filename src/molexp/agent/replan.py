"""Replan API — sibling-Run authoring driven by an agent's verdict.

Pairs with :meth:`Workflow.sanity_check`: when a sanity hook fires
``on_fail='replan'``, the orchestrator (a coding-agent driving the run,
or any external supervisor) calls :func:`replan` with a *modifier* that
maps the offending run's parameters to a corrected dict. ``replan``
materialises a fresh sibling :class:`molexp.workspace.Run` under the
same experiment, records ``replanned_from`` + ``replanned_reason`` for
provenance, and returns the new run for the orchestrator to start.

This module sits in :mod:`molexp.agent` rather than
:mod:`molexp.workflow` deliberately — workflow stays a pure DAG; the
replan loop is an *agent-side* control plane that consumes workflow
sanity events and emits new workspace Runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from molexp.workspace.run import Run


def replan(
    run: Run,
    *,
    modifier: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    reason: str | None = None,
) -> Run:
    """Materialise a sibling :class:`Run` whose parameters reflect ``modifier``.

    Args:
        run: The original run that triggered the replan — typically the
            run whose ``sanity_check(on_fail='replan')`` hook fired.
        modifier: A callable that receives the original run's parameter
            mapping and returns the new parameter dict.  The returned
            mapping replaces the original parameters wholesale (it does
            not deep-merge); callers that want to preserve unmodified
            keys should spread them in:
            ``lambda p: {**p, "k": v}``.
        reason: Optional human-readable note pinned on the new run's
            metadata labels under ``replanned_reason``.

    Returns:
        The freshly materialised :class:`Run` instance.  The original
        run is left untouched.
    """
    original_params = dict(run.parameters)
    new_params = dict(modifier(original_params))

    new_run = run.experiment.run(parameters=new_params)
    labels = dict(new_run.metadata.labels)
    labels["replanned_from"] = run.id
    if reason is not None:
        labels["replanned_reason"] = reason
    new_run._update_metadata(labels=labels)
    return new_run


__all__ = ["replan"]
