"""``ChatMode`` — harness-level chat orchestration (peer of Plan).

Chat uses the same :class:`~molexp.agent.loops.interactive.InteractiveLoop` as
Plan, but a **different tool surface and land policy**:

* default **no** authoritative project/run creation
* **no** ``run_land`` (non-standard products must not pollute the workspace)
* code confined to ``agent/.scratch/``
* success = answer + optional scratch scripts, not a succeeded Run

Authoritative multi-step workflows remain :class:`PlanOrchestrator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.harness.schemas import ModeResult

if TYPE_CHECKING:
    from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig
    from molexp.agent.session import Session
    from molexp.harness.gateways.gateway import AgentGateway

__all__ = ["CHAT_SCRATCH_PREFIX", "ChatMode", "chat_loop_config"]

CHAT_SCRATCH_PREFIX = "agent/.scratch"


def chat_loop_config(
    *,
    workspace_root: Path | None = None,
    context_block: str = "",
    system_prompt: str = "",
) -> InteractiveLoopConfig:
    """Build :class:`~molexp.agent.loops.InteractiveLoopConfig` for Chat Mode.

    Always ``operation_mode="chat"`` (scratch-only builtins, no default land).
    """
    from molexp.agent.loops import InteractiveLoopConfig

    return InteractiveLoopConfig(
        workspace_root=workspace_root,
        context_block=context_block,
        system_prompt=system_prompt,
        operation_mode="chat",
    )


class ChatMode:
    """Harness Chat Mode — InteractiveLoop, no default workspace land.

    Attributes:
        name: Mode id (``"chat"``), peer of Plan's ``"plan"``.
    """

    name = "chat"

    def build_loop(
        self,
        *,
        workspace_root: Path | None = None,
        context_block: str = "",
        system_prompt: str = "",
        hooks: object | None = None,
        tools: tuple[object, ...] = (),
    ) -> InteractiveLoop:
        """Construct the chat :class:`InteractiveLoop` (scratch surface)."""
        from molexp.agent.loops import InteractiveLoop
        from molexp.agent.loops.hooks import LoopHooks

        return InteractiveLoop(
            config=chat_loop_config(
                workspace_root=workspace_root,
                context_block=context_block,
                system_prompt=system_prompt,
            ),
            hooks=hooks if isinstance(hooks, LoopHooks) else None,  # type: ignore[arg-type]
            tools=tools,
        )

    async def run(
        self,
        *,
        workspace_root: Path,
        user_input: str,
        gateway: AgentGateway,
        session: Session | None = None,
        context_block: str = "",
    ) -> ModeResult:
        """Drive one chat turn via InteractiveLoop; do not create a science Run.

        Unlike Plan, chat does not materialize content-addressed experiment
        runs. Session messages may still persist under the agent folder.
        """
        from molexp.agent.events import AsyncIteratorEventSink
        from molexp.agent.execution_env import LocalExecutionEnv
        from molexp.agent.runtime import AgentRuntime
        from molexp.agent.session import InMemorySessionStorage, Session

        router = getattr(gateway, "router", None)
        if router is None:
            raise TypeError(
                "ChatMode requires gateway.router (RouterBackedAgentGateway or test fake)"
            )
        root = Path(workspace_root).resolve()
        scratch = root / CHAT_SCRATCH_PREFIX
        scratch.mkdir(parents=True, exist_ok=True)

        sess = session if session is not None else Session(storage=InMemorySessionStorage())
        runtime = AgentRuntime(
            session=sess,
            router=router,  # type: ignore[arg-type]
            execution_env=LocalExecutionEnv(scratch_dir=scratch),
        )
        loop = self.build_loop(workspace_root=root, context_block=context_block)
        sink = AsyncIteratorEventSink()
        try:
            await loop.run(runtime=runtime, sink=sink, user_input=user_input)
        finally:
            await sink.close()

        # Chat has no content-addressed science Run; ids are session placeholders.
        return ModeResult(
            mode_name=self.name,
            run_id="chat",
            execution_id="chat",
        )
