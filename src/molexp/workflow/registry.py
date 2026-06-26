"""Task-type registry for IR-driven workflow assembly.

The agent emits a :term:`workflow IR` (JSON) where each node carries a
``task_type`` slug like ``"core.add"``. This registry maps slugs to
factories that instantiate the corresponding :class:`Task` (or any
:class:`~Runnable`).

Slugs are deliberately *not* Python FQNs — the same workflow JSON should
load on a server that ships the same plugin set even if module paths
shift, and the UI has no business handling Python paths.

Usage
-----

Register a task type::

    from molexp.workflow.registry import default_registry
    from molexp.workflow.task import Task


    class Add(Task):
        async def execute(self, ctx) -> float:
            inputs = ctx._inputs or {}
            return sum(float(v) for v in inputs.values())


    default_registry.register("core.add", Add)

Look it up::

    factory = default_registry.get("core.add")
    instance = factory({})  # config dict goes to constructor kwargs
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from .context import TaskContext
from .protocols import JSONMapping, TaskBody, TaskInput, TaskOutput
from .task import Task

# A factory takes a config dict and returns a Runnable / Streamable / callable.
type TaskFactory = Callable[[JSONMapping], TaskBody]

# Either a ready-made factory or a class whose ``__init__`` accepts the
# config dict's keys as kwargs. The class branch is wrapped on registration
# into a factory via ``lambda cfg: cls(**cfg)``. ``type`` matches any class
# at the static-typing level; the runtime narrows via ``isinstance(_, type)``.
type RegistrableTarget = TaskFactory | type


class TaskTypeRegistry:
    """Maps task-type slugs to factories that produce executable task objects.

    A factory receives the IR's ``config`` dict and returns:

    - a :class:`~molexp.workflow.protocols.Runnable` (object with ``async execute()``)
    - a :class:`~molexp.workflow.protocols.Streamable` (object with ``async run()``)
    - a bare async callable

    Slugs are case-sensitive, dotted (``"<namespace>.<name>"``).
    """

    def __init__(self) -> None:
        self._factories: dict[str, TaskFactory] = {}
        self._descriptions: dict[str, str] = {}
        # Reverse index: task class -> slug. Populated only for *class*
        # registrations (a bare factory callable has no single class to map
        # back from). This is what lets serialization resolve a task's slug
        # from its type, so authors never restate it at ``add()`` time.
        self._slug_for_class: dict[type, str] = {}

    @overload
    def register(
        self,
        slug: str,
        factory: None = None,
        *,
        description: str = "",
    ) -> Callable[[RegistrableTarget], RegistrableTarget]: ...

    @overload
    def register(
        self,
        slug: str,
        factory: RegistrableTarget,
        *,
        description: str = "",
    ) -> RegistrableTarget: ...

    def register(
        self,
        slug: str,
        factory: RegistrableTarget | None = None,
        *,
        description: str = "",
    ) -> Callable[[RegistrableTarget], RegistrableTarget] | RegistrableTarget:
        """Register ``slug`` → ``factory``.

        ``factory`` may be:

        - A callable taking ``config: dict`` and returning the task instance.
        - A class whose ``__init__`` accepts keyword arguments matching
          ``config`` (the registry wraps it as ``lambda cfg: cls(**cfg)``).

        When called as a decorator (``factory=None``), returns the wrapper
        so usage like ``@registry.register("core.add")\\nclass Add(Task): ...``
        works.
        """

        def _wrap(target: RegistrableTarget) -> RegistrableTarget:
            if isinstance(target, type):
                cls = target

                def _from_class(cfg: JSONMapping) -> TaskBody:
                    return cls(**(cfg or {}))

                self._factories[slug] = _from_class
                self._bind_reverse(cls, slug)
            else:
                self._factories[slug] = target
            if description:
                self._descriptions[slug] = description
            return target

        if factory is None:
            return _wrap
        return _wrap(factory)

    def _bind_reverse(self, cls: type, slug: str) -> None:
        """Record ``cls -> slug`` for serialization-time resolution.

        One slug per class: re-registering the same class under the same slug
        is idempotent, but a second, *different* slug for an already-mapped
        class is a configuration error (the reverse lookup would be ambiguous).
        """
        existing = self._slug_for_class.get(cls)
        if existing is not None and existing != slug:
            raise ValueError(
                f"{cls.__qualname__} is already registered as {existing!r}; "
                f"cannot also register it as {slug!r} (one slug per task type)."
            )
        self._slug_for_class[cls] = slug

    def slug_for(self, target: object) -> str | None:
        """Return the registry slug for a task body / class, or ``None``.

        Accepts a task *instance* (resolves via ``type(target)``) or a class
        directly. Returns ``None`` when the type was never registered, or was
        registered only as a bare factory callable (no class to map back from).
        This is the inverse of :meth:`get` and the reason ``WorkflowCompiler.add``
        needs no ``task_type`` argument.
        """
        cls = target if isinstance(target, type) else type(target)
        return self._slug_for_class.get(cls)

    def get(self, slug: str) -> TaskFactory:
        """Return the factory for ``slug``, raising on unknown slugs."""
        try:
            return self._factories[slug]
        except KeyError as exc:
            known = ", ".join(sorted(self._factories)) or "<empty>"
            raise KeyError(f"Unknown task_type {slug!r}. Registered types: {known}") from exc

    def has(self, slug: str) -> bool:
        return slug in self._factories

    def slugs(self) -> list[str]:
        """Return all registered slugs in deterministic order."""
        return sorted(self._factories)

    def describe(self, slug: str) -> str:
        return self._descriptions.get(slug, "")

    def items(self) -> list[tuple[str, str]]:
        """Return ``(slug, description)`` pairs in deterministic order."""
        return [(s, self._descriptions.get(s, "")) for s in self.slugs()]


# ── Module singleton ───────────────────────────────────────────────────────

default_registry = TaskTypeRegistry()


# ── Built-in demo tasks ────────────────────────────────────────────────────
# Tiny, numeric, deterministic — primarily for testing the IR round-trip and
# giving the agent a non-trivial composable starter set. Domain plugins
# (LAMMPS, VASP, molpy) register their own slugs through the same registry.


class _Constant(Task):
    """Emit a fixed value. Useful as a graph root in tests / demos."""

    def __init__(self, value: TaskOutput = 0) -> None:
        self.value = value

    async def execute(self, ctx: TaskContext) -> TaskOutput:  # noqa: ARG002
        return self.value


class _Add(Task):
    """Sum the numeric outputs of all upstream tasks.

    Accepts the standard input shapes produced by the runtime:
    ``None`` (no upstream → 0), a single value, or a ``dict[str, value]``.
    """

    async def execute(self, ctx: TaskContext) -> float:
        return _coerce_sum(ctx._inputs)


class _Multiply(Task):
    """Multiply the single upstream value by ``factor``."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = float(factor)

    async def execute(self, ctx: TaskContext) -> float:
        if ctx._inputs is None:
            return 0.0
        return float(ctx._inputs) * self.factor


def _coerce_sum(inputs: TaskInput) -> float:
    if inputs is None:
        return 0.0
    if isinstance(inputs, dict):
        return sum(float(v) for v in inputs.values())
    if isinstance(inputs, (list, tuple)):
        return sum(float(v) for v in inputs)
    return float(inputs)


default_registry.register(
    "core.constant",
    _Constant,
    description="Emit a fixed value (config: {value: any}).",
)
default_registry.register(
    "core.add",
    _Add,
    description="Sum numeric outputs of upstream tasks.",
)
default_registry.register(
    "core.multiply",
    _Multiply,
    description="Multiply single upstream value by `factor` (config: {factor: float}).",
)


__all__ = [
    "TaskFactory",
    "TaskTypeRegistry",
    "default_registry",
]
