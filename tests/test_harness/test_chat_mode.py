"""Chat — one-shot structured AgentCall, scratch-only, no default land."""

from __future__ import annotations

from pathlib import Path

from molexp.harness import Chat, chat_loop_config
from molexp.harness.modes.chat import CHAT_SCRATCH_PREFIX


def test_chat_loop_config_is_chat_surface() -> None:
    cfg = chat_loop_config(workspace_root=Path("/tmp/ws"))
    assert cfg.operation_mode == "chat"
    assert cfg.workspace_root == Path("/tmp/ws")


def test_chat_name() -> None:
    assert Chat().name == "chat"


def test_chat_scratch_prefix() -> None:
    assert CHAT_SCRATCH_PREFIX == "agent/.scratch"


def test_chat_plugins_apply_on_run(tmp_path: Path) -> None:
    import asyncio
    from datetime import UTC, datetime

    from molexp.harness.host.context import Context
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec, PlanArtifactRef

    applied: list[str] = []

    class _Marker:
        name = "marker"
        inject: tuple[str, ...] = ()

        def apply(self, ctx: Context) -> None:
            applied.append("yes")
            ctx.provide("marker", True)

    class _Gw:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            del spec
            dummy = PlanArtifactRef(
                id="art-1",
                kind="log",
                uri="memory://art-1",
                sha256="a" * 64,
                created_at=datetime.now(tz=UTC),
                created_by="t",
            )
            return AgentCallResult(
                output_artifact=dummy,
                raw_response_artifact=dummy,
                model="none",
            )

        def register_agent(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    asyncio.run(
        Chat(plugins=(_Marker(),)).run(
            workspace_root=tmp_path,
            user_input="hi",
            gateway=_Gw(),  # type: ignore[arg-type]
        )
    )
    assert applied == ["yes"]
