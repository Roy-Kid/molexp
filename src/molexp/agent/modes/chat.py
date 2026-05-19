"""``ChatMode`` — single-turn LLM round-trip wrapped as a 1-task workflow.

Even a one-shot LLM call is a workflow now: ``ChatMode.run()`` builds a
single-task :class:`Workflow` and awaits its :meth:`Workflow.execute`,
keeping the agent layer uniformly workflow-driven. The wrapping is
deliberate — it removes the bespoke "just call the router" branch and
gives chat mode the same persistence / observability hooks every other
mode gets.

Multi-turn support is unchanged: the session's ``model_messages``
field carries the pydantic-ai-native ``ModelMessage`` history back
into ``Agent.run(message_history=...)`` on every turn so the LLM sees
the full conversation context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.router import ModelTier
from molexp.agent.types import Message
from molexp.workflow import Task, TaskContext, Workflow, WorkflowBuilder

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


_LOG = get_logger(__name__)


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


@dataclass(frozen=True)
class _ChatDeps:
    """Runtime services for the chat workflow's single task."""

    router: Router
    system_prompt: str
    message_history: tuple[Any, ...]
    tier: ModelTier


@dataclass(frozen=True)
class _ChatTurnResult:
    """Output of :class:`ChatTurn` — surfaced into ``WorkflowResult.outputs``."""

    text: str
    raw: Any


class ChatTurn(Task):
    """The chat workflow's only task. One router call, one structured return."""

    async def execute(
        self, ctx: TaskContext[None, _ChatDeps, None]
    ) -> _ChatTurnResult:
        user_input = ctx.config.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("ChatTurn requires a non-empty 'user_input' in config.")
        result = await ctx.deps.router.complete_text(
            prompt=user_input,
            system=ctx.deps.system_prompt,
            message_history=ctx.deps.message_history,
            tier=ctx.deps.tier,
        )
        return _ChatTurnResult(text=result.text, raw=result.raw)


def build_chat_workflow() -> Workflow:
    """Assemble the 1-task chat workflow.

    Returned :class:`Workflow` is frozen and content-addressed; one
    instance lives at module scope so repeat calls reuse the same
    ``workflow_id``.
    """
    builder = WorkflowBuilder(name="chat_mode", entry="ChatTurn")
    builder.add(ChatTurn(), name="ChatTurn")
    return builder.build()


CHAT_WORKFLOW: Workflow = build_chat_workflow()
"""Module-level frozen workflow for ChatMode."""


class ChatMode(AgentMode):
    """One ``user_input`` → one workflow execution → one :class:`AgentRunResult`."""

    name = "chat"

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        router.clear_usage()
        session.append(Message(role="user", content=user_input))

        deps = _ChatDeps(
            router=router,
            system_prompt=self.config.system_prompt,
            message_history=tuple(session.model_messages),
            tier=ModelTier.DEFAULT,
        )

        result = await CHAT_WORKFLOW.execute(
            config={"user_input": user_input},
            deps=deps,
        )
        turn = result.outputs.get("ChatTurn")
        if not isinstance(turn, _ChatTurnResult):
            raise RuntimeError(
                f"ChatMode: workflow {CHAT_WORKFLOW.name!r} returned status={result.status!r} "
                "without a ChatTurn output."
            )

        session.append(Message(role="assistant", content=turn.text))
        session.model_messages = _extract_all_messages(turn.raw, session.model_messages)
        breakdown = router.snapshot_usage()
        _LOG.info(
            f"[chat-mode] usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} total={breakdown.total.total_tokens} "
            f"reqs={breakdown.total.requests}"
        )
        return AgentRunResult(
            text=turn.text,
            messages=tuple(session.history),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )


def _extract_all_messages(
    raw: Any,  # noqa: ANN401 — opaque pydantic-ai RunResult; the agent layer firewall
    fallback: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Pull the cumulative pydantic-ai message list off a ``RunResult``.

    pydantic-ai's ``AgentRunResult.all_messages()`` returns the full
    conversation including the latest turn — the canonical value to
    pass back as ``message_history`` next time. Stub routers (used by
    tests) leave ``raw`` empty / shapeless; we degrade to the existing
    history so callers can still chain turns deterministically.
    """
    if raw is None:
        return fallback
    getter = getattr(raw, "all_messages", None)
    if not callable(getter):
        return fallback
    return tuple(getter())


__all__ = [
    "CHAT_WORKFLOW",
    "ChatMode",
    "ChatModeConfig",
    "ChatTurn",
    "build_chat_workflow",
]
