"""``AgentMode`` ABC + ``AgentRunResult`` value type.

A mode encodes the strategy: ChatMode does a single LLM round-trip;
the four pipeline modes (Plan / Author / Run / Review, specs 03-06)
drive multi-stage workflows. Every mode runs *on* an
:class:`~molexp.agent.harness.harness.AgentHarness` — the shared runtime
that owns the :class:`~molexp.agent.harness.session.Session`, the
:data:`~molexp.agent.harness.events.AgentEvent` stream, compaction, the
:class:`~molexp.agent.harness.execution_env.ExecutionEnv`, and the hook
registry.

:meth:`AgentMode.run` is an async generator: it *yields*
:data:`AgentEvent`\\ s as the mode progresses, and its terminal yield
is a :class:`~molexp.agent.harness.events.ModeCompletedEvent` carrying
the final :class:`AgentRunResult`. :class:`~molexp.agent.runner.AgentRunner`
drains the stream, accumulates it, and returns the terminal result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, Usage, UsageBreakdown

if TYPE_CHECKING:
    from molexp.agent.harness.events import AgentEvent
    from molexp.agent.harness.harness import AgentHarness


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    Modes populate ``mode_state`` with mode-specific structured output
    (a plan, a review verdict, …); ChatMode leaves it ``None``.

    ``usage`` is the aggregate token / request count for the run;
    ``usage_breakdown`` is the per-call list (one entry per LLM round
    trip). Both default empty when no LLM call is made.

    ``events`` holds the accumulated orchestration-level
    :data:`~molexp.agent.harness.events.AgentEvent` stream the mode
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

    Pure declarative metadata — describes how a mode's stages connect,
    not how they execute.

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


class ModePipeline(BaseModel):
    """The declarative stage topology of an :class:`AgentMode`.

    A mode is a hand-written async sequence of ``harness.stage(...)``
    brackets — there is no ``Graph`` object. ``ModePipeline`` is the
    *declarative* mirror of that topology: pure side-band metadata that
    powers :meth:`AgentMode.get_flowchart` and is held honest against
    the real ``run()`` source by the per-mode no-drift test. It never
    participates in execution.

    Attributes:
        stages: Ordered stage names, one per ``harness.stage("...")``
            call in the mode's ``run()`` source.
        edges: Labelled control-flow edges (branches and repair loops).
        terminal_states: Names of the run's terminal states.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    stages: tuple[str, ...] = ()
    edges: tuple[PipelineEdge, ...] = ()
    terminal_states: tuple[str, ...] = ()


def _mermaid_node_id(name: str) -> str:
    """Coerce a stage / terminal name into a Mermaid-safe node id.

    Non-alphanumeric characters become ``_`` so a hyphenated stage name
    like ``chat-turn`` yields a valid identifier (``chat_turn``).
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    return cleaned or "_"


def _render_pipeline_flowchart(pipeline: ModePipeline) -> str:
    """Render a :class:`ModePipeline` as Mermaid ``flowchart TD`` text.

    The agent layer's own tiny renderer — it does **not** import
    ``molexp.workflow``'s Mermaid code (cross-layer presentation-helper
    imports are forbidden) nor ``pydantic_graph``. Stages render as
    rectangle nodes, terminal states as stadium nodes, edges as
    ``-->`` (labelled edges as ``-->|label|``).
    """
    lines = ["flowchart TD"]
    for stage in pipeline.stages:
        lines.append(f'    {_mermaid_node_id(stage)}["{stage}"]')
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
        :class:`~molexp.agent.harness.events.ModeCompletedEvent` whose
        ``result`` carries the JSON dump of the terminal
        :class:`AgentRunResult`.
        """
        ...

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
    from molexp.agent.harness.events import AgentEvent as _AgentEvent

    AgentRunResult.model_rebuild(_types_namespace={"AgentEvent": _AgentEvent})


_rebuild_models()


__all__ = ["AgentMode", "AgentRunResult", "ModePipeline", "PipelineEdge"]
