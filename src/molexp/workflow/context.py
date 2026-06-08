"""Execution context for workflow tasks and streaming actors.

``TaskContext`` is the **single object** every user-defined task (batch
``Task`` or streaming ``Actor``) receives. After the pure-task-context collapse
it carries:

* ``inputs`` — the runtime data flowing in along the graph's edges (upstream
  task outputs; for a root task, whatever the engine injects: the run's sweep
  params and a content-addressed working-directory ``Path``);
* ``config`` — the build-time static configuration declared at ``add()`` time
  (part of the node's content identity);
* ``state`` — **read-only** shared workflow state (``results`` etc.).

There is **no** ``run_context`` and **no** ``deps``: a task cannot climb up from
its context to the Run, the workspace, or injected services. Engine capabilities
that used to be reached through ``run_context`` — a content-addressed workdir,
artifact persistence, running a sub-workflow — are delivered *as inputs* by the
engine, or handled by the engine's materialization layer after the body returns.

``state`` is retained **deliberately and minimally** (decision #2): fully
removing it would break loop/branch accumulation and routed-value access, which
read prior/routed outputs from ``state.results``. Delivering those via ``inputs``
needs engine-level loop-back/routed input wiring, deferred to its own spec
(``pure-task-context-*-state-elimination``). Until then ``state`` stays as a
read-only escape hatch for those patterns.

The context is frozen: it is a plain class (NOT a pydantic model — ``inputs``
carries arbitrary live task outputs such as numpy arrays or PyO3 objects that a
pydantic model would try to validate/copy; per CLAUDE.md live-value containers
are plain classes). Attribute assignment raises.
"""

from __future__ import annotations

from .protocols import JSONMapping


class TaskContext[StateT, InputT]:
    """Frozen context passed to every ``Task.execute()`` / ``Actor.run()``.

    Attributes:
        inputs: Runtime data flowing in along the edges — upstream task outputs
            (``None`` for a root task with nothing injected), or the engine's
            injected inputs for a root task (sweep params + a workdir ``Path``).
        config: Read-only build-time configuration mapping (defaults to ``{}``).
        state: Read-only shared workflow state (retained minimally for loop /
            branch data-flow; see module docstring).
    """

    _inputs: InputT
    _config: JSONMapping
    _state: StateT | None
    __slots__ = ("_config", "_inputs", "_state")

    def __init__(
        self,
        inputs: InputT,
        config: JSONMapping | None = None,
        state: StateT | None = None,
    ) -> None:
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_config", config if config is not None else {})
        object.__setattr__(self, "_state", state)

    @property
    def inputs(self) -> InputT:
        """Runtime data flowing in along the edges."""
        return self._inputs

    @property
    def config(self) -> JSONMapping:
        """Read-only mapping of build-time configuration."""
        return self._config

    @property
    def state(self) -> StateT | None:
        """Read-only shared workflow state (loop / branch data-flow)."""
        return self._state

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"TaskContext is frozen; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"TaskContext(inputs={self._inputs!r}, config={self._config!r})"
