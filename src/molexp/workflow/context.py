"""Execution contexts for workflow tasks and actors.

``TaskContext`` is the **single object** every user-defined task receives.
It carries five attributes — ``state``, ``deps``, ``inputs``, ``config``,
``run_context`` — and nothing else. Workspace plumbing (artifacts, logs,
checkpoints, named results) is no longer the workflow layer's concern;
callers attach those capabilities through the opaque duck-typed
``run_context`` payload and reach into them directly when needed.
"""

from __future__ import annotations

from .protocols import JSONMapping, RunContextLike, TaskOutput


class TaskContext[StateT, DepsT, InputT]:
    """Context passed to every ``Task.execute()``.

    Attributes:
        state: Shared mutable workflow state visible to all tasks.
        deps: Injected dependencies (any user object).
        inputs: Typed output from the upstream task (``None`` for root tasks).
        config: JSON-shaped mapping exposed to the task body (defaults to ``{}``).
        run_context: Duck-typed run context (``RunContextLike``) supplied by
            the caller of ``WorkflowSpec.execute(run_context=...)``. Tasks
            that need workspace capabilities reach for them through this
            object; the workflow layer holds it via the structural Protocol.
    """

    def __init__(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
        config: JSONMapping | None = None,
        run_context: RunContextLike | None = None,
    ) -> None:
        self._state = state
        self._deps = deps
        self._inputs = inputs
        self._config: JSONMapping = config if config is not None else {}
        self._run_ctx = run_context

    @property
    def state(self) -> StateT:
        """Shared mutable workflow state visible to all tasks."""
        return self._state

    @property
    def deps(self) -> DepsT:
        """Injected dependencies (any user object)."""
        return self._deps

    @property
    def inputs(self) -> InputT:
        """Typed output from the upstream task (``None`` for root tasks)."""
        return self._inputs

    @property
    def run_context(self) -> RunContextLike | None:
        """Duck-typed run-context payload supplied by the caller (or ``None``)."""
        return self._run_ctx

    @property
    def config(self) -> JSONMapping:
        """Read-only mapping of user-supplied configuration."""
        return self._config


class ActorContext[StateT, DepsT, InputT](TaskContext[StateT, DepsT, InputT]):
    """Extended context for streaming ``Actor`` tasks.

    Adds async message-passing primitives on top of ``TaskContext``.
    """

    async def receive(self) -> InputT:
        """Wait for the next message from upstream.

        Requires the ``run_context`` payload to expose a ``receive(name)``
        coroutine; otherwise raises :class:`NotImplementedError`.
        """
        if self._run_ctx is not None and hasattr(self._run_ctx, "receive"):
            return await self._run_ctx.receive("input")
        raise NotImplementedError("receive() requires a run_context with .receive(name)")

    async def send(self, output: TaskOutput) -> None:
        """Send a message to downstream actors.

        Requires the ``run_context`` payload to expose an ``emit(name, value)``
        coroutine; otherwise raises :class:`NotImplementedError`.
        """
        if self._run_ctx is not None and hasattr(self._run_ctx, "emit"):
            await self._run_ctx.emit("output", output)
            return
        raise NotImplementedError("send() requires a run_context with .emit(name, value)")
