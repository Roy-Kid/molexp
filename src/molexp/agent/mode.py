"""``AgentMode`` ABC + ``AgentRunResult`` value type + executable ``ModePipeline``.

A mode encodes the strategy: ChatMode does a single LLM round-trip;
the four pipeline modes (Plan / Author / Run / Review, specs 03-06)
drive multi-stage workflows. Every mode runs *on* an
:class:`~molexp.agent.runtime.AgentHarness` — the shared runtime
that owns the :class:`~molexp.agent.session.Session`, the
:data:`~molexp.agent.events.AgentEvent` stream, compaction, the
:class:`~molexp.agent.execution_env.ExecutionEnv`, and the hook
registry.

:meth:`AgentMode.run` is an async generator: it *yields*
:data:`AgentEvent`\\ s as the mode progresses, and its terminal yield
is a :class:`~molexp.agent.events.ModeCompletedEvent` carrying
the final :class:`AgentRunResult`. :class:`~molexp.agent.runner.AgentRunner`
drains the stream, accumulates it, and returns the terminal result.

:class:`ModePipeline` is the **executable IR** the substrate built in
spec ``agent-mode-stage-pipeline-01`` introduces: a plain Python
container (not a pydantic BaseModel — it carries live
:class:`~molexp.agent.stage.Stage` instances) listing the
mode's stages, the typed control-flow edges, the terminal-state
names, the entry-stage name, and any
:class:`~molexp.agent.repair.RepairPolicy`s. Subclasses set
their ``pipeline`` once at class-body level and may delegate ``run``
to :meth:`run_pipeline` (a thin wrapper around the substrate's
:func:`~molexp.agent.pipeline.execute_pipeline`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, Usage, UsageBreakdown

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent
    from molexp.agent.repair import RepairPolicy
    from molexp.agent.runtime import AgentHarness
    from molexp.agent.stage import Stage


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    Modes populate ``mode_state`` with mode-specific structured output
    (a plan, a review verdict, …); ChatMode leaves it ``None``.

    ``usage`` is the aggregate token / request count for the run;
    ``usage_breakdown`` is the per-call list (one entry per LLM round
    trip). Both default empty when no LLM call is made.

    ``events`` holds the accumulated orchestration-level
    :data:`~molexp.agent.events.AgentEvent` stream the mode
    emitted while running — it defaults to ``()`` so callers that only
    want the terminal text are unaffected. All other fields are
    unchanged from the pre-harness contract.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    text: str
    messages: tuple[Message, ...] = ()
    mode_state: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    usage_breakdown: UsageBreakdown = Field(default_factory=UsageBreakdown)
    events: tuple[AgentEvent, ...] = ()


class PipelineEdge(BaseModel):
    """One labelled control-flow edge of a :class:`ModePipeline`.

    Pure declarative data — describes how a mode's stages connect.

    Attributes:
        from_stage: Source stage name.
        to_stage: Target stage name, or a terminal-state name.
        label: Optional edge label (a branch condition, e.g.
            ``"pass"`` / ``"rejected"``); ``None`` for an unlabelled edge.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    from_stage: str
    to_stage: str
    label: str | None = None


class ModePipeline:
    """The executable stage topology of an :class:`AgentMode`.

    A plain Python class (not pydantic — Stage instances and the
    optional ``lifecycle_validator`` carry live runtime references,
    and ``arbitrary_types_allowed=True`` is forbidden under
    ``src/molexp/agent/``).

    Constructor accepts:

    Attributes:
        stages: Tuple of :class:`~molexp.agent.stage.Stage`
            instances making up the pipeline (order is informational
            — execution follows :attr:`edges`).
        edges: Tuple of :class:`PipelineEdge`s describing control
            flow between stages and into terminal states.
        terminal_states: Names of the run's terminal states.
        entry: Name of the first stage to execute. Defaults to the
            first stage's name when stages are supplied.
        repairs: Tuple of :class:`~molexp.agent.repair.RepairPolicy`
            instances; the executor honours these in declaration
            order.
        lifecycle_validator: Optional callable invoked once per stage
            entry (before the stage's ``run`` body) with
            ``(stage, harness)``. Modes that maintain a typed
            lifecycle (e.g. ``PlanState``) plug a translator here;
            the harness itself stays unaware of mode-specific state
            machines.
    """

    def __init__(
        self,
        *,
        stages: tuple[Stage, ...] = (),
        edges: tuple[PipelineEdge, ...] = (),
        terminal_states: tuple[str, ...] = (),
        entry: str = "",
        repairs: tuple[RepairPolicy, ...] = (),
        lifecycle_validator: Callable[[Stage, AgentHarness], None] | None = None,
    ) -> None:
        self.stages = stages
        self.edges = edges
        self.terminal_states = terminal_states
        self.entry = entry if entry else (stages[0].name if stages else "")
        self.repairs = repairs
        self.lifecycle_validator = lifecycle_validator


def _mermaid_node_id(name: str) -> str:
    """Coerce a stage / terminal name into a Mermaid-safe node id."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    return cleaned or "_"


def _render_pipeline_flowchart(pipeline: ModePipeline) -> str:
    """Render a :class:`ModePipeline` as Mermaid ``flowchart TD`` text.

    The agent layer's own tiny renderer — it does **not** import
    ``molexp.workflow``'s Mermaid code (cross-layer presentation-helper
    imports are forbidden) nor ``pydantic_graph``. Stages render as
    rectangle nodes (the name comes from each Stage's ``name``),
    terminal states as stadium nodes, edges as ``-->`` (labelled
    edges as ``-->|label|``).
    """
    lines = ["flowchart TD"]
    for stage in pipeline.stages:
        lines.append(f'    {_mermaid_node_id(stage.name)}["{stage.name}"]')
    for terminal in pipeline.terminal_states:
        lines.append(f'    {_mermaid_node_id(terminal)}(["{terminal}"])')
    for edge in pipeline.edges:
        src = _mermaid_node_id(edge.from_stage)
        dst = _mermaid_node_id(edge.to_stage)
        if edge.label:
            lines.append(f"    {src} -->|{edge.label}| {dst}")
        else:
            lines.append(f"    {src} --> {dst}")
    return "\n".join(lines) + "\n"


class AgentMode(ABC):
    """Abstract strategy a mode must implement to be drivable by ``AgentRunner``.

    Subclasses set ``name`` to a stable identifier and implement
    :meth:`run` as an async generator. The ``harness`` keyword is
    supplied by :class:`~molexp.agent.runner.AgentRunner`; user code
    does not call ``run`` directly.

    Resume contract
    ---------------
    :meth:`resume` is a classmethod that reconstructs a mode instance
    from persisted state. The default raises :exc:`NotImplementedError`;
    subclasses override it to read their own on-disk format.

    Pipeline delegation
    -------------------
    :meth:`run_pipeline` is a concrete helper that drains
    :func:`~molexp.agent.pipeline.execute_pipeline` on
    ``self.pipeline``. Subclasses migrating to the new substrate may
    delegate their ``run`` body to it with ``async for ev in
    self.run_pipeline(harness=harness, user_input=user_input): yield ev``.
    No mode delegates yet in phase 01 — phases 02 and 03 of the chain
    do the migration.
    """

    name: str = ""
    pipeline: ModePipeline = ModePipeline()

    @abstractmethod
    def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the mode, yielding orchestration events as it progresses.

        The final yield must be a
        :class:`~molexp.agent.events.ModeCompletedEvent` whose
        ``result`` carries the JSON dump of the terminal
        :class:`AgentRunResult`.
        """
        ...

    async def run_pipeline(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
        initial_input: object | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Default delegation helper — drain ``self.pipeline`` via the substrate.

        Subclasses migrating to the new abstraction route their
        ``run`` body through this helper. ``initial_input`` defaults
        to ``user_input`` for simplicity; modes that need a typed
        entry payload can pass it explicitly.
        """
        from molexp.agent.pipeline import execute_pipeline

        payload = user_input if initial_input is None else initial_input
        async for event in execute_pipeline(
            pipeline=self.pipeline,
            harness=harness,
            user_input=user_input,
            initial_input=payload,
        ):
            yield event

    def get_flowchart(self) -> str:
        """Return this mode's stage pipeline as a Mermaid ``flowchart TD``.

        Renders the declarative :attr:`pipeline` — every concrete mode
        overrides ``pipeline`` with its own :class:`ModePipeline`. The
        rendering is pure metadata: it does not run the mode and does
        not touch the harness.
        """
        return _render_pipeline_flowchart(self.pipeline)

    @classmethod
    def resume(cls, **kwargs: Any) -> AgentMode:  # noqa: ANN401
        """Reconstruct this mode from persisted state.

        Subclasses override this to read their own on-disk format.
        The default raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError(f"{cls.__name__} does not support resume")


# Resolve the forward reference to AgentEvent so AgentRunResult can be
# validated/serialized at runtime (the field type is only TYPE_CHECKING
# imported above to keep the module's import graph shallow).
def _rebuild_models() -> None:
    """Inject ``AgentEvent`` and rebuild :class:`AgentRunResult`."""
    from molexp.agent.events import AgentEvent as _AgentEvent

    AgentRunResult.model_rebuild(_types_namespace={"AgentEvent": _AgentEvent})


_rebuild_models()


__all__ = ["AgentMode", "AgentRunResult", "ModePipeline", "PipelineEdge"]
