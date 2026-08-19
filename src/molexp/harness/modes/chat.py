"""``Chat`` — one-shot chat bundle. Must not loop.

Chat is one ``AgentGateway.call`` with ``call_mode="structured"``.
Tool-using REPL turns belong on :class:`~molexp.agent.AgentRunner`
(``mode="agentic"``). Authoritative multi-step work is :class:`Plan`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.harness.schemas import ModeResult

if TYPE_CHECKING:
    from molexp.agent.session import Session
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.host.plugin import Plugin

__all__ = ["CHAT_SCRATCH_PREFIX", "Chat", "ChatConfig", "ChatReply", "chat_loop_config"]

CHAT_SCRATCH_PREFIX = "agent/.scratch"


class ChatReply(BaseModel):
    """One-shot chat payload — a single assistant reply, no tool loop."""

    model_config = ConfigDict(frozen=True)

    text: str = ""


@dataclass(frozen=True)
class ChatConfig:
    """Scratch-only chat settings (one-shot, no session loop)."""

    workspace_root: Path | None = None
    context_block: str = ""
    system_prompt: str = ""
    operation_mode: str = "chat"


def chat_loop_config(
    *,
    workspace_root: Path | None = None,
    context_block: str = "",
    system_prompt: str = "",
) -> ChatConfig:
    """Build :class:`ChatConfig` for Chat Mode (always ``operation_mode="chat"``)."""
    return ChatConfig(
        workspace_root=workspace_root,
        context_block=context_block,
        system_prompt=system_prompt,
        operation_mode="chat",
    )


class Chat:
    """One structured AgentCall. Options on the instance; ``run`` takes the turn."""

    name = "chat"

    def __init__(
        self,
        *,
        context_block: str = "",
        plugins: tuple[Plugin, ...] = (),
    ) -> None:
        self.context_block = context_block
        self.plugins = plugins

    async def run(
        self,
        *,
        workspace_root: Path,
        user_input: str,
        gateway: AgentGateway,
        session: Session | None = None,
        context_block: str = "",
    ) -> ModeResult:
        """Drive one chat turn through :meth:`AgentGateway.call` (no loop)."""
        del session  # one-shot; session persistence is the AgentRunner REPL's job
        from molexp.agent.router import ModelTier
        from molexp.harness.gateways.gateway import AgentGateway as Gateway
        from molexp.harness.host.compose import compose_chat
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.artifact_store import ArtifactStore

        block = context_block or self.context_block
        root = Path(workspace_root).resolve()
        scratch = root / CHAT_SCRATCH_PREFIX
        scratch.mkdir(parents=True, exist_ok=True)
        register = getattr(gateway, "register_agent", None)
        if callable(register):
            register(
                "chat",
                ChatReply,
                "assistant_message",
                tier=ModelTier.DEFAULT,
                system_prompt=block,
            )
        host = compose_chat(gateway=gateway, scratch_dir=scratch, extra=self.plugins)
        try:
            atom = host.ctx.llm
            store = host.ctx.artifacts
            if not isinstance(atom, Gateway):
                raise TypeError("chat host did not publish an AgentGateway")
            if not isinstance(store, ArtifactStore):
                raise TypeError("chat host did not publish an ArtifactStore")
            prompt_ref = store.put_text(
                kind="prompt",
                text=user_input,
                created_by="chat",
                parent_ids=[],
            )
            await atom.call(
                AgentCallSpec(
                    agent_name="chat",
                    input_artifact_ids=[prompt_ref.id],
                    output_schema=ChatReply.model_json_schema(),
                    call_mode="structured",
                ),
            )
        finally:
            host.unload()

        return ModeResult(
            mode_name=self.name,
            run_id="chat",
            execution_id="chat",
        )
