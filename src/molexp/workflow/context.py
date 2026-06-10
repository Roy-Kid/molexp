"""Execution context for workflow tasks and streaming actors.

``TaskContext`` is the **single object** every user-defined task (batch
``Task`` or streaming ``Actor``) receives. After the pure-task-context collapse
it carries:

* ``inputs`` — the runtime data flowing in along the graph's edges (upstream
  task outputs; for a root task, whatever the engine injects: the run's sweep
  params and a content-addressed working-directory ``Path``);
* ``config`` — the build-time static configuration declared at ``add()`` time
  (part of the node's content identity);
* ``state`` — **DEPRECATED** (staged removal): emits a ``DeprecationWarning``
  and returns a read-only snapshot.

There is **no** ``run_context`` and **no** ``deps``: a task cannot climb up from
its context to the Run, the workspace, or injected services. Engine capabilities
that used to be reached through ``run_context`` — a content-addressed workdir,
artifact persistence, running a sub-workflow — are delivered *as inputs* by the
engine, or handled by the engine's materialization layer after the body returns.

``state`` is in **staged removal** (pure-task-context state-elimination): the
values-on-edges engine now delivers loop-back and branch-routed values via
``ctx.inputs`` (declared ``depends_on`` wins; trigger-carried values reach
dep-less targets), so the patterns that used to read ``state.results`` no
longer need it. Accessing ``ctx.state`` emits a ``DeprecationWarning`` and
returns a READ-ONLY snapshot of the underlying state (a ``MappingProxyType``
copy for mappings; a frozen :class:`ReadOnlyStateView` for engine
``WorkflowState``-shaped objects) — user code can still read legacy values but
can no longer mutate engine state. Hard removal is the remaining step.

The context is frozen: it is a plain class (NOT a pydantic model — ``inputs``
carries arbitrary live task outputs such as numpy arrays or PyO3 objects that a
pydantic model would try to validate/copy; per CLAUDE.md live-value containers
are plain classes). Attribute assignment raises.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

from .protocols import JSONMapping

if TYPE_CHECKING:
    from pathlib import Path


class _WorkflowStateLike(Protocol):
    """Duck-typed read surface of the engine's ``WorkflowState``.

    Only ``results`` is required; the remaining fields are read defensively
    via ``getattr`` so any state-shaped object qualifies.
    """

    results: dict[str, Any]


_STATE_DEPRECATION_MSG = (
    "TaskContext.state is deprecated: values now arrive via ctx.inputs "
    "(the values-on-edges engine delivers loop-back and branch-routed values "
    "on the activating edge); ctx.state will be removed."
)


class ReadOnlyStateView:
    """Frozen, point-in-time snapshot of an engine ``WorkflowState``'s read surface.

    Built lazily by :attr:`TaskContext.state` so legacy ``ctx.state.results``
    reads keep returning correct values during the deprecation window, while
    mutation of engine state through the context is impossible: ``results`` is
    a ``MappingProxyType`` over a *copy*, ``completed`` / ``seeded`` are
    ``frozenset``s, and attribute assignment raises.
    """

    _completed: frozenset[str]
    _error: str | None
    _failed: bool
    _results: Mapping[str, Any]
    _seeded: frozenset[str]
    __slots__ = ("_completed", "_error", "_failed", "_results", "_seeded")

    def __init__(self, state: _WorkflowStateLike) -> None:
        object.__setattr__(self, "_results", MappingProxyType(dict(state.results)))
        object.__setattr__(self, "_completed", frozenset(getattr(state, "completed", ())))
        object.__setattr__(self, "_seeded", frozenset(getattr(state, "seeded", ())))
        object.__setattr__(self, "_failed", bool(getattr(state, "failed", False)))
        object.__setattr__(self, "_error", getattr(state, "error", None))

    @property
    def results(self) -> Mapping[str, Any]:
        """Read-only ``task_name → output`` snapshot (copy at access time)."""
        return self._results

    @property
    def completed(self) -> frozenset[str]:
        """Names of tasks that finished at least once (snapshot)."""
        return self._completed

    @property
    def seeded(self) -> frozenset[str]:
        """Names seeded via ``execute(seed_outputs=...)`` (snapshot)."""
        return self._seeded

    @property
    def failed(self) -> bool:
        """Terminal failure flag (snapshot)."""
        return self._failed

    @property
    def error(self) -> str | None:
        """Terminal failure message (snapshot)."""
        return self._error

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"ReadOnlyStateView is frozen; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"ReadOnlyStateView(results={dict(self._results)!r})"


def _freeze_state[StateT](state: StateT) -> StateT | ReadOnlyStateView | Mapping[Any, Any] | None:
    """Build the read-only snapshot :attr:`TaskContext.state` returns.

    * ``None`` → ``None``;
    * a ``Mapping`` → ``MappingProxyType`` over a copy;
    * an engine ``WorkflowState``-shaped object (dict ``results`` attribute)
      → :class:`ReadOnlyStateView`;
    * anything else is returned unchanged (opaque user object — the
      deprecation warning still fires).
    """
    if state is None:
        return None
    if isinstance(state, Mapping):
        return MappingProxyType(dict(state))
    if isinstance(getattr(state, "results", None), dict):
        return ReadOnlyStateView(cast("_WorkflowStateLike", state))
    return state


class TaskContext[StateT, InputT]:
    """Frozen context passed to every ``Task.execute()`` / ``Actor.run()``.

    Attributes:
        inputs: Runtime data flowing in along the edges — upstream task outputs
            (``None`` for a root task with nothing injected), or the engine's
            injected inputs for a root task (sweep params + a workdir ``Path``).
        config: Read-only build-time configuration mapping (defaults to ``{}``).
        state: DEPRECATED — emits a ``DeprecationWarning`` and returns a
            read-only snapshot; loop / branch values now arrive via ``inputs``
            (see module docstring).
        workdir: Content-addressed scratch directory for THIS task — a bare
            ``pathlib.Path`` the engine derives from the task's content identity
            (its ``TaskSnapshot.key``) via the materialization layer. It is the
            sanctioned place a task writes intermediate files (a task body that
            does ``ctx.workdir / name`` gets a stable, per-task location reused
            across runs). ``None`` when no materialization layer is active (e.g. a
            plain non-workspace run). A fan-out body shares one ``workdir`` across
            elements, so per-element bodies should sub-namespace it.
    """

    _inputs: InputT
    _config: JSONMapping
    _state: StateT | None
    _workdir: Path | None
    __slots__ = ("_config", "_inputs", "_state", "_workdir")

    def __init__(
        self,
        inputs: InputT,
        config: JSONMapping | None = None,
        state: StateT | None = None,
        workdir: Path | None = None,
    ) -> None:
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_config", config if config is not None else {})
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_workdir", workdir)

    @property
    def inputs(self) -> InputT:
        """Runtime data flowing in along the edges."""
        return self._inputs

    @property
    def workdir(self) -> Path | None:
        """Content-addressed scratch directory for this task (``None`` if absent)."""
        return self._workdir

    @property
    def config(self) -> JSONMapping:
        """Read-only mapping of build-time configuration."""
        return self._config

    @property
    def state(self) -> StateT | ReadOnlyStateView | Mapping[Any, Any] | None:
        """DEPRECATED read-only snapshot of shared workflow state.

        Emits a ``DeprecationWarning`` on every access: loop-back and
        branch-routed values now arrive via ``ctx.inputs``; ``ctx.state``
        will be removed. The returned object is a point-in-time, read-only
        snapshot (see :func:`_freeze_state`) — engine state cannot be
        mutated through it.
        """
        warnings.warn(_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return _freeze_state(self._state)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"TaskContext is frozen; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"TaskContext(inputs={self._inputs!r}, config={self._config!r})"
