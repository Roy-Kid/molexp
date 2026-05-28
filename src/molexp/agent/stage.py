"""The ``Stage`` first-class abstraction.

A :class:`Stage` is one unit of work in an
:class:`~molexp.agent.mode.AgentMode`'s pipeline. The harness's
:func:`~molexp.agent.pipeline.execute_pipeline` walks a
:class:`~molexp.agent.mode.ModePipeline`'s tuple of Stage instances,
brackets each one in :meth:`AgentHarness.stage` and drains the stage's
async-generator :meth:`Stage.run` body.

Subclasses define their stage by setting the ``name`` class variable
and implementing :meth:`run` as an async generator. Lifecycle tagging
(``pre_state`` / ``post_state``) is opaque to the harness — modes that
care about lifecycle interpret these tags via the optional
``lifecycle_validator`` on their :class:`~molexp.agent.mode.ModePipeline`.

This is a **plain Python class** — not a ``pydantic.BaseModel`` —
because Stage subclasses may carry callables, async iterators, and
service handles. ``arbitrary_types_allowed=True`` is forbidden under
``src/molexp/agent/`` per the agent-layer charter, so the substrate
refuses to be pydantic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent
    from molexp.agent.runtime import AgentHarness

__all__ = ["NameOnlyStage", "Stage"]


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(Generic[InputT, OutputT], ABC):  # noqa: UP046 — keep classic Generic for ty / runtime introspection
    """One unit of work in a mode pipeline.

    Subclasses pin :attr:`name` (a stable identifier) and implement
    :meth:`run` as an async generator. The generator yields
    :data:`~molexp.agent.events.AgentEvent` instances as the
    stage progresses; the *final* yielded value (whatever its type) is
    treated as the stage's typed output and threaded as the next
    stage's ``input``.

    The three optional ClassVars carry **opaque** lifecycle tags the
    harness never interprets:

    - :attr:`pre_state` — what mode-specific state the stage assumes
      on entry; ``None`` means "no precondition".
    - :attr:`post_state` — what mode-specific state the stage leaves
      the run in on success; ``None`` means "no transition".
    - :attr:`persists` — names of artefacts this stage writes (used by
      a mode's lifecycle_validator to coordinate workspace
      persistence).
    """

    name: ClassVar[str]
    pre_state: ClassVar[str | None] = None
    post_state: ClassVar[str | None] = None
    persists: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(
        self,
        *,
        harness: AgentHarness,
        input: InputT,
    ) -> AsyncIterator[AgentEvent | OutputT]:
        """Drive this stage, yielding orchestration events.

        Implemented by subclasses as ``async def run(...)`` with at
        least one ``yield``. Yield :data:`AgentEvent` instances as the
        stage progresses; the *final* yielded value is the stage's
        terminal output and becomes the next stage's ``input``.

        ``harness`` is the live runtime; ``input`` is the previous
        stage's terminal output (or the pipeline's ``initial_input``
        for the entry stage).
        """
        ...


class NameOnlyStage(Stage[object, object]):
    """Transitional Stage placeholder carrying only a name.

    Phase-01 substrate companion: lets pre-migration modes (chat /
    plan / author / run / review / interactive) keep their
    ``pipeline = ModePipeline(stages=(NameOnlyStage("X"), ...), ...)``
    declarations with the new Stage-instance shape *without* migrating
    each mode's ``run()`` body to drive
    :func:`~molexp.agent.pipeline.execute_pipeline`.

    Phase 02 + 03 of the chain replace each use of this with a real
    Stage subclass. At chain-end the class is deleted.

    Its :meth:`run` raises :exc:`NotImplementedError` — this stage is
    declarative-only and is never meant to be drained during phase 01
    (no mode delegates to ``run_pipeline`` yet).
    """

    def __init__(self, name: str) -> None:
        # Override the ClassVar ``Stage.name`` with an instance attribute
        # — Python permits the per-instance override at runtime. ty's
        # ``invalid-attribute-access`` diagnostic is intentional and is
        # suppressed only for this transitional placeholder; the class
        # is deleted at chain end so the suppression goes with it.
        self.name = name  # ty: ignore[invalid-attribute-access]

    async def run(  # type: ignore[override] # noqa: ANN201
        self,
        *,
        harness,  # noqa: ANN001, ARG002 — kept for ABC compatibility
        input,  # noqa: ANN001, ARG002 — kept for ABC compatibility
    ):
        raise NotImplementedError(
            f"NameOnlyStage({self.name!r}) is a transitional placeholder "
            "for the phase-01 substrate; it cannot execute. The owning "
            "mode has not been migrated to drive run_pipeline yet."
        )
        if False:  # pragma: no cover — keeps this an async generator
            yield
