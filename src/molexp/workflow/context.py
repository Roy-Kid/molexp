"""Execution context for workflow tasks and streaming actors.

``TaskContext`` is the **single object** every user-defined task (batch
``Task`` or streaming ``Actor``) receives. After the pure-task-context collapse
it carries:

* ``inputs`` — the runtime data flowing in along the graph's edges (upstream
  task outputs; for a root task, whatever the engine injects: the run's sweep
  params and a content-addressed working-directory ``Path``);
* ``config`` — the build-time static configuration declared at ``add()`` time
  (part of the node's content identity).

There is **no** ``run_context``, **no** ``deps``, and **no** ``state``: a task
cannot climb up from its context to the Run, the workspace, or engine
workflow state. Loop-back and branch-routed values arrive as named task
parameters (declared ``depends_on`` wins; trigger-carried values reach
dep-less targets). Engine capabilities that used to be reached through
``run_context`` — a content-addressed workdir, artifact persistence, running
a sub-workflow — are delivered *as inputs* by the engine, or handled by the
engine's materialization layer after the body returns.

The context is frozen: it is a plain class (NOT a pydantic model — ``inputs``
carries arbitrary live task outputs such as numpy arrays or PyO3 objects that a
pydantic model would try to validate/copy; per CLAUDE.md live-value containers
are plain classes). Attribute assignment raises.
"""

from __future__ import annotations

import json
from pathlib import Path

from .outputs import RegisterArtifact, RegisterMetric
from .protocols import JSONMapping


class TaskContext[StateT, InputT]:
    """Frozen context passed to every ``Task.execute()`` / ``Actor.run()``.

    Runtime inputs are **not** read off the context — they bind to the body's own
    typed parameters by name (``async def task(ctx, sigma: float = 1.0)``; the
    engine fills ``sigma`` from {upstream task outputs keyed by task name} |
    {run sweep params} | {build config}, falling back to the declared default).
    ``ctx`` exposes ``workdir`` plus the deferred ``register_artifact`` /
    ``register_metric`` verbs (there is no ``ctx.inputs`` / ``ctx.config``).

    Attributes:
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
    _workdir: Path | None
    _pending: list[RegisterArtifact | RegisterMetric]
    __slots__ = ("_config", "_inputs", "_pending", "_workdir")

    def __init__(
        self,
        inputs: InputT,
        config: JSONMapping | None = None,
        workdir: Path | None = None,
    ) -> None:
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_config", config if config is not None else {})
        object.__setattr__(self, "_workdir", workdir)
        object.__setattr__(self, "_pending", [])

    @property
    def workdir(self) -> Path | None:
        """Content-addressed scratch directory for this task (``None`` if absent).

        Runtime inputs are not reached through the context — they bind to the
        body's own typed parameters by name. The former ``ctx.inputs`` /
        ``ctx.config`` are gone from the public surface; the underlying values
        survive as private slots only so engine-internal adapters can read them.
        """
        return self._workdir

    def register_artifact(
        self,
        data: Path | bytes | dict | list | str,
        *,
        name: str | None = None,
        mime: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> RegisterArtifact:
        """Record intent to publish *data* as a run artifact.

        Does **not** touch the Run. The engine promotes the returned
        :class:`RegisterArtifact` after the body returns. In-memory payloads
        are written under :attr:`workdir` first (``name`` required).
        """
        path = data if isinstance(data, Path) else self._write_pending_file(data, name)
        marker = RegisterArtifact(
            path,
            name=name if name is not None else path.name,
            tags=tags,
            mime=mime,
        )
        self._pending.append(marker)
        return marker

    def register_metric(
        self,
        key: str,
        value: float,
        *,
        step: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> RegisterMetric:
        """Record intent to append a scalar metric on the run.

        Does **not** touch the Run. The engine promotes the returned
        :class:`RegisterMetric` after the body returns.
        """
        marker = RegisterMetric(key, value, step=step, tags=tags)
        self._pending.append(marker)
        return marker

    def _write_pending_file(
        self,
        data: bytes | dict | list | str,
        name: str | None,
    ) -> Path:
        if name is None:
            raise ValueError("register_artifact: name is required when data is not a Path")
        if self._workdir is None:
            raise RuntimeError(
                "register_artifact: in-memory data requires ctx.workdir; "
                "pass a Path or run under a workspace"
            )
        target = self._workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            target.write_bytes(bytes(data))
        elif isinstance(data, (dict, list)):
            target.write_text(json.dumps(data, indent=2, default=str))
        else:
            target.write_text(str(data))
        return target

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"TaskContext is frozen; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"TaskContext(inputs={self._inputs!r}, config={self._config!r})"
