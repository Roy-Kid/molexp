"""Public agent surface — pydantic-ai facade.

The user-visible surface is four names — :class:`AgentRunner`,
:class:`AgentRunResult`, :class:`AgentRuntime`, :class:`AgentSession` —
plus one **lazy** re-export: ``PydanticAIRouter``.

There is no molexp-owned conversation loop. Chat is one
``Router.complete_text``. Tool-using work is one ReAct
(``Router.stream_agentic``). Plan orchestration is a
:class:`~molexp.workflow.WorkflowCompiler` graph in harness.

Layer position: **agent uses workspace only**. It **MUST NOT** import
:mod:`molexp.workflow`, :mod:`molexp.harness`, or any sibling
application layer. The harness imports agent via the sanctioned
``agent.router`` Protocol edge.

``import molexp.agent`` does not eagerly load ``pydantic_ai``.
"""

from typing import TYPE_CHECKING

from molexp.agent.loop import AgentRunResult
from molexp.agent.runner import AgentRunner
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session as AgentSession

if TYPE_CHECKING:
    from molexp.agent._pydanticai.router import PydanticAIRouter as PydanticAIRouter

__all__ = [
    "AgentRunResult",
    "AgentRunner",
    "AgentRuntime",
    "AgentSession",
]


def __getattr__(name: str) -> object:
    """Lazy re-export — keeps ``import molexp.agent`` pydantic-ai-free."""
    if name == "PydanticAIRouter":
        from molexp.agent._pydanticai.router import PydanticAIRouter

        return PydanticAIRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
