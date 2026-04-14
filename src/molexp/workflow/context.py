"""Execution contexts for workflow tasks and actors.

``TaskContext`` is the **single object** every user-defined task receives.
It bridges the workflow type system (state, deps, inputs) with optional
workspace capabilities (artifacts, assets, checkpoints).

When a workflow runs outside a workspace (pure computation mode), the
workspace methods gracefully return ``None`` instead of raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic

from molexp.config import ProfileConfig

from .types import DepsT, InputT, StateT


class TaskContext(Generic[StateT, DepsT, InputT]):
    """Context passed to every ``Task.execute()``.

    Workflow-side (always available):
        - ``state``   — shared mutable workflow state
        - ``deps``    — injected dependencies
        - ``inputs``  — typed output from the upstream task
        - ``config``  — active :class:`~molexp.config.ProfileConfig`

    Workspace-side (available when ``run`` is provided):
        - ``save_artifact()``
        - ``get_artifact_path()``
        - ``find_asset()``
        - ``checkpoint()``
        - ``set_result()`` / ``get_result()``
    """

    def __init__(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
        config: ProfileConfig | None = None,
        run_context: Any | None = None,
    ) -> None:
        self._state = state
        self._deps = deps
        self._inputs = inputs
        self._config = config if config is not None else ProfileConfig({}, name=None)
        self._run_ctx = run_context  # workspace.run.RunContext or None

    # ── Workflow-side properties ─────────────────────────────────────────

    @property
    def state(self) -> StateT:
        """Shared mutable workflow state visible to all tasks."""
        return self._state

    @property
    def deps(self) -> DepsT:
        """Injected dependencies (workspace, external services, …)."""
        return self._deps

    @property
    def inputs(self) -> InputT:
        """Typed output from the upstream task (``None`` for root tasks)."""
        return self._inputs

    @property
    def run_context(self) -> Any:
        """Access the underlying RunContext (``None`` if no run attached)."""
        return self._run_ctx

    @property
    def config(self) -> ProfileConfig:
        """Active molcfg profile configuration (read-only mapping)."""
        return self._config

    # ── Workspace-side helpers ───────────────────────────────────────────
    # These return None / no-op when running without a workspace Run.

    def save_artifact(self, name: str, data: Any) -> Path | None:
        """Persist an artifact to the run's artifact directory.

        Returns the written path, or ``None`` if no run is attached.
        """
        if self._run_ctx is None:
            return None
        return self._run_ctx.save_artifact(name, data)

    def get_artifact_path(self, name: str) -> Path | None:
        """Return the path to a previously saved artifact."""
        if self._run_ctx is None:
            return None
        return self._run_ctx.get_artifact_path(name)

    def find_asset(self, name: str) -> Any:
        """Search for an asset up the scope hierarchy.

        Order: experiment -> project -> workspace.
        Returns ``None`` if not found or no run is attached.
        """
        if self._run_ctx is None:
            return None
        return self._run_ctx.find_asset(name)

    def checkpoint(self, name: str | None = None) -> str | None:
        """Create a checkpoint of current execution state.

        Returns the checkpoint ID, or ``None`` if no run is attached.
        """
        if self._run_ctx is None:
            return None
        return self._run_ctx.checkpoint(name)

    def set_result(self, key: str, value: Any) -> None:
        """Store a named result in the run context."""
        if self._run_ctx is not None:
            self._run_ctx.set_result(key, value)

    def get_result(self, key: str) -> Any:
        """Retrieve a previously stored result."""
        if self._run_ctx is None:
            return None
        return self._run_ctx.get_result(key)


class ActorContext(TaskContext[StateT, DepsT, InputT]):
    """Extended context for streaming ``Actor`` tasks.

    Adds async message-passing primitives on top of ``TaskContext``.
    """

    async def receive(self) -> InputT:
        """Wait for the next message from upstream."""
        if self._run_ctx is not None and hasattr(self._run_ctx, "receive"):
            return await self._run_ctx.receive("input")
        raise NotImplementedError("receive() requires a connected run context")

    async def send(self, output: Any) -> None:
        """Send a message to downstream actors."""
        if self._run_ctx is not None and hasattr(self._run_ctx, "emit"):
            await self._run_ctx.emit("output", output)
            return
        raise NotImplementedError("send() requires a connected run context")
