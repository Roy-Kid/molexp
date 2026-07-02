"""Dependent-params resolution for per-task node bodies.

When a task declares ``dependent_params=fn``, its config is computed from the
upstream tasks' outputs (and assets) at run time. This module owns that
resolution and the small upstream-view proxies it hands to ``fn`` — kept apart
from the dispatch core in :mod:`.node`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..protocols import (
    AssetsViewLike,
    JSONMapping,
    RunContextLike,
    TaskInput,
    TaskOutput,
)
from .state import WorkflowState

if TYPE_CHECKING:
    from .._graph_decl import TaskRegistration


class _UpstreamView:
    """Per-upstream view passed to ``dependent_params(prev)``.

    Exposes ``.output`` (the upstream task's return value, as recorded in
    :attr:`WorkflowState.results`) and ``.assets`` (an
    :class:`~molexp.workspace.assets.AssetsView` filtered to the upstream
    task's producer entries when a workspace ``RunContext`` is attached;
    ``None`` otherwise).
    """

    __slots__ = ("assets", "output")

    def __init__(self, output: TaskOutput, assets: _UpstreamAssetsView | None) -> None:
        self.output = output
        self.assets = assets


def _resolve_dependent_params(
    *,
    registration: TaskRegistration,
    state: WorkflowState,
    run_context: RunContextLike | None,
    base_config: JSONMapping | None,
) -> JSONMapping | None:
    """If the task declares ``dependent_params=fn``, resolve and overlay onto config.

    ``fn`` receives ``dict[str, _UpstreamView]`` keyed by upstream task name.
    Its return mapping is overlayed onto a fresh
    :class:`~molexp.profile.ProfileConfig` and the result replaces the task's
    base config. The base config is returned unchanged when no
    ``dependent_params`` is declared.
    """
    fn = getattr(registration, "dependent_params", None)
    if fn is None:
        return base_config

    from molexp.profile import ProfileConfig

    prev: dict[str, _UpstreamView] = {}
    for dep in registration.depends_on:
        upstream_assets = None
        if run_context is not None:
            assets_view = getattr(run_context, "assets", None)
            if assets_view is not None and hasattr(assets_view, "query"):
                upstream_assets = _UpstreamAssetsView(assets_view, producer_task=dep)
        prev[dep] = _UpstreamView(
            output=state.results.get(dep),
            assets=upstream_assets,
        )

    overlay = fn(prev)
    if overlay is None:
        return base_config
    if not isinstance(overlay, Mapping):
        raise TypeError(
            f"dependent_params for task {registration.name!r} must return a Mapping; "
            f"got {type(overlay).__name__}"
        )
    merged: dict[str, TaskInput] = dict(base_config) if base_config is not None else {}
    merged.update(overlay)
    return ProfileConfig(merged, name=getattr(base_config, "name", None))


class _UpstreamAssetsView:
    """Lazy ``query()`` proxy that pre-binds ``producer_task=<dep>``.

    Avoids importing :class:`AssetsView` at module top to keep the
    workspace dependency optional for non-workspace runs.
    """

    __slots__ = ("_inner", "_producer_task")

    def __init__(self, assets_view: AssetsViewLike, producer_task: str) -> None:
        self._inner = assets_view
        self._producer_task = producer_task

    def query(
        self,
        *,
        kind: str | type | None = None,
        producer_run: str | None = None,
        producer_task: str | None = None,
        tag: tuple[str, str] | None = None,
        limit: int | None = None,
        recursive: bool = False,
    ) -> TaskOutput:
        return self._inner.query(
            kind=kind,
            producer_run=producer_run,
            producer_task=producer_task or self._producer_task,
            tag=tag,
            limit=limit,
            recursive=recursive,
        )

    def list(self) -> TaskOutput:
        return self.query()
