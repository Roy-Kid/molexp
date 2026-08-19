"""Plugin host: Context, inject, unload, AgentCall waterfall, plan/chat compose."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.errors import AgentCallRejectedError, PluginInjectError
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.host import (
    Context,
    Host,
    Keys,
    compose_chat,
    compose_curate,
    compose_plan,
    compose_run,
)
from molexp.harness.host.plugins.agent_call import AgentCallPlugin
from molexp.harness.schemas import AgentCallResult, AgentCallSpec, PlanArtifactRef
from molexp.harness.store.file_artifact_store import FileArtifactStore


class _NeedsArtifacts:
    name = "needs_artifacts"
    inject = (Keys.ARTIFACTS,)

    def apply(self, ctx: Context) -> None:
        ctx.provide("marker", ctx.require(Keys.ARTIFACTS))


class _FakeRouter:
    async def complete_text(self, **kwargs: object) -> object:
        raise NotImplementedError

    async def complete_structured(self, **kwargs: object) -> object:
        raise NotImplementedError

    def stream_agentic(self, **kwargs: object) -> object:
        raise NotImplementedError


class _FakeGateway:
    def __init__(self) -> None:
        self.router = _FakeRouter()
        self.calls: list[AgentCallSpec] = []

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        self.calls.append(spec)
        dummy = PlanArtifactRef(
            id="art-1",
            kind="log",
            uri="memory://art-1",
            sha256="a" * 64,
            created_at=datetime.now(tz=UTC),
            created_by="test",
        )
        return AgentCallResult(
            output_artifact=dummy,
            raw_response_artifact=dummy,
            model="fake",
        )


class TestContextEffects:
    def test_provide_and_require(self) -> None:
        ctx = Context()
        ctx.provide("k", 1)
        assert ctx.require("k") == 1
        assert ctx.has("k")

    def test_provide_rejects_duplicate(self) -> None:
        ctx = Context()
        ctx.provide("k", 1)
        with pytest.raises(ValueError, match="already provided"):
            ctx.provide("k", 2)

    def test_require_missing_fails_loud(self) -> None:
        ctx = Context()
        with pytest.raises(KeyError, match="not on the host"):
            ctx.require("missing")

    def test_unwind_drops_provided_key(self) -> None:
        ctx = Context()
        mark = ctx.effect_count()
        ctx.provide("k", 1)
        ctx.unwind_to(mark)
        assert not ctx.has("k")


class TestContextAttr:
    def test_provided_service_is_an_attribute(self) -> None:
        belt = object()
        ctx = Context()
        ctx.provide("tools", belt)
        assert ctx.tools is belt

    def test_missing_service_raises_attribute_error(self) -> None:
        ctx = Context()
        with pytest.raises(AttributeError, match="jobs"):
            _ = ctx.jobs
        assert hasattr(ctx, "jobs") is False

    def test_methods_are_not_shadowed(self) -> None:
        ctx = Context()
        assert callable(ctx.provide)
        assert hasattr(ctx, "provide") is True


class TestHostLifecycle:
    def test_missing_inject_fails_at_mount(self) -> None:
        host = Host()
        with pytest.raises(PluginInjectError) as exc:
            host.mount(_NeedsArtifacts())
        assert exc.value.plugin == "needs_artifacts"
        assert Keys.ARTIFACTS in exc.value.missing

    def test_agent_call_requires_artifacts(self) -> None:
        host = Host()
        with pytest.raises(PluginInjectError) as exc:
            host.mount(AgentCallPlugin(_FakeGateway()))
        assert Keys.ARTIFACTS in exc.value.missing

    def test_mount_order_is_dump(self) -> None:
        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        host.mount(AgentCallPlugin(_FakeGateway()))
        assert host.dump() == ["llm"]

    def test_unload_removes_service(self) -> None:
        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        host.mount(AgentCallPlugin(_FakeGateway()))
        assert host.ctx.has(Keys.LLM)
        host.unload("llm")
        assert not host.ctx.has(Keys.LLM)
        assert host.dump() == []


class TestWaterfall:
    async def test_pre_call_rewrites_spec(self) -> None:
        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))

        async def rewrite(spec: object, nxt: object) -> object:
            assert isinstance(spec, AgentCallSpec)
            rewritten = spec.model_copy(update={"agent_name": "rewritten"})
            return await nxt(rewritten)  # type: ignore[misc,operator]

        host.ctx.on("agent/pre-step", rewrite, mode="waterfall")
        atom = host.ctx.llm
        result = await atom.call(  # type: ignore[union-attr]
            AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
        )
        assert inner.calls[0].agent_name == "rewritten"
        assert result.model == "fake"

    async def test_pre_call_reject_skips_inner(self) -> None:
        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))

        async def reject(spec: object, nxt: object) -> object:
            del spec, nxt
            raise AgentCallRejectedError("denied")

        host.ctx.on("agent/pre-step", reject, mode="waterfall")
        atom = host.ctx.llm
        with pytest.raises(AgentCallRejectedError, match="denied"):
            await atom.call(  # type: ignore[union-attr]
                AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
            )
        assert inner.calls == []

    async def test_retired_pre_call_name_does_not_run(self) -> None:
        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))
        ran = {"n": 0}

        async def stale(spec: object, nxt: object) -> object:
            ran["n"] += 1
            return await nxt(spec)  # type: ignore[misc,operator]

        host.ctx.on("agent/pre-call", stale, mode="waterfall")
        atom = host.ctx.llm
        await atom.call(  # type: ignore[union-attr]
            AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
        )
        assert ran["n"] == 0
        assert inner.calls[0].agent_name == "orig"

    async def test_post_step_waterfall_can_replace_result(self) -> None:
        from molexp.harness.host.plugins.agent_call import AgentStep

        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))

        async def swap(step: object, nxt: object) -> object:
            current = await nxt(step)  # type: ignore[misc,operator]
            assert isinstance(current, AgentStep)
            return AgentStep(
                spec=current.spec,
                result=current.result.model_copy(update={"model": "swapped"}),
            )

        host.ctx.on("agent/post-step", swap, mode="waterfall")
        atom = host.ctx.llm
        result = await atom.call(  # type: ignore[union-attr]
            AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
        )
        assert result.model == "swapped"

    def test_sources_spell_pre_step_not_pre_call(self) -> None:
        import inspect

        from molexp.harness import errors
        from molexp.harness.host.plugins import agent_call

        err_src = inspect.getsource(errors.AgentCallRejectedError)
        call_src = inspect.getsource(agent_call)
        assert "pre-step" in err_src
        assert "pre-call" not in err_src
        assert "pre-step" in call_src
        assert "pre-call" not in call_src


class TestReflection:
    async def test_critic_replaces_result(self) -> None:
        from molexp.harness.host.plugins.agent_call import AgentStep
        from molexp.harness.host.plugins.reflection import Reflection

        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))

        async def critic(step: AgentStep) -> object:
            return step.result.model_copy(update={"model": "reflected"})

        host.mount(Reflection(critic=critic))
        atom = host.ctx.llm
        result = await atom.call(  # type: ignore[union-attr]
            AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
        )
        assert result.model == "reflected"

    async def test_skip_list_does_not_call_critic(self) -> None:
        from molexp.harness.host.plugins.agent_call import AgentStep
        from molexp.harness.host.plugins.reflection import Reflection

        host = Host()
        host.ctx.provide(Keys.ARTIFACTS, object())
        inner = _FakeGateway()
        host.mount(AgentCallPlugin(inner))
        called = {"n": 0}

        async def critic(step: AgentStep) -> object:
            del step
            called["n"] += 1
            return None

        host.mount(Reflection(critic=critic, skip=("chat",)))
        atom = host.ctx.llm
        result = await atom.call(  # type: ignore[union-attr]
            AgentCallSpec(agent_name="chat", input_artifact_ids=[], output_schema={})
        )
        assert called["n"] == 0
        assert result.model == "fake"


class TestKeys:
    def test_spine_names_match_deepseek(self) -> None:
        assert Keys.LLM == "llm"
        assert Keys.TOOLS == "tools"
        assert Keys.FS == "fs"
        assert Keys.APPROVAL == "approval"
        assert Keys.SESSIONS == "sessions"
        assert Keys.SYSTEM_PROMPT == "systemPrompt"
        assert Keys.JOBS == "jobs"
        assert Keys.SANDBOX == "sandbox"
        assert Keys.COMMANDS == "commands"
        assert Keys.SETTINGS == "settings"
        assert Keys.CREDENTIALS == "credentials"
        assert Keys.WORKSPACE == "workspace"
        assert Keys.WORKFLOW == "workflow"
        assert not hasattr(Keys, "AGENT_LOOP")
        assert not hasattr(Keys, "AGENT_CALL")


class TestImportStayLight:
    def test_import_host_does_not_pull_workflow_or_sdk(self) -> None:
        probe = (
            "import sys, importlib;"
            "importlib.import_module('molexp.harness.host');"
            "forbidden = ['molexp.workflow', 'pydantic_ai', 'pydantic_graph'];"
            "loaded = [m for m in forbidden if m in sys.modules];"
            "print('LOADED:' + ','.join(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = result.stdout.strip().removeprefix("LOADED:")
        assert loaded == ""


class TestCompose:
    def test_compose_chat_mounts_llm(self, tmp_path: Path) -> None:
        host = compose_chat(gateway=_FakeGateway(), scratch_dir=tmp_path)
        assert host.dump() == ["run_stores", "tools", "llm"]
        assert host.ctx.has(Keys.LLM)
        assert host.ctx.has(Keys.TOOLS)
        assert hasattr(host.ctx, "llm")
        assert not hasattr(host.ctx, "agent_call")
        host.unload()
        assert host.dump() == []

    def test_compose_plan_projects_run_context(self, tmp_path: Path) -> None:
        host = compose_plan(
            run_id="abcd1234",
            run_dir=tmp_path,
            gateway=_FakeGateway(),
        )
        assert host.dump() == [
            "run_stores",
            "tools",
            "approval_policy",
            "workspace",
            "workflow",
            "llm",
        ]
        assert host.ctx.tools is host.ctx.require(Keys.TOOLS)
        cfg = host.dump_config()
        assert "tools" in cfg["services"]
        assert "artifacts" in cfg["services"]
        ctx = host.as_run_context()
        assert ctx.run_id == "abcd1234"
        assert ctx.workspace_root == tmp_path
        assert isinstance(ctx.artifact_store, FileArtifactStore)
        assert ctx.agent_gateway is not None
        assert not hasattr(ctx, "host")

    def test_compose_curate_uses_workspace_root_not_run_dir(self, tmp_path: Path) -> None:
        ws_root = tmp_path / "ws"
        run_dir = tmp_path / "run"
        ws_root.mkdir()
        run_dir.mkdir()
        host = compose_curate(
            run_id="abcd1234",
            run_dir=run_dir,
            workspace_root=ws_root,
        )
        assert host.dump() == ["run_stores"]
        ctx = host.as_run_context()
        assert ctx.workspace_root == ws_root
        assert ctx.agent_gateway is None

    def test_compose_plan_binds_gateway_store(self, tmp_path: Path) -> None:
        other = FileArtifactStore(root=tmp_path / "other")
        gw = StubAgentGateway(other)
        host = compose_plan(run_id="abcd1234", run_dir=tmp_path, gateway=gw)
        ctx = host.as_run_context()
        assert gw.artifact_store is ctx.artifact_store

    def test_compose_run_mounts_executor_and_workflow(self, tmp_path: Path) -> None:
        host = compose_run(run_id="abcd1234", run_dir=tmp_path)
        assert host.dump() == ["run_stores", "executor", "workspace", "workflow"]
        assert host.ctx.has(Keys.EXECUTOR)
        assert host.ctx.has(Keys.WORKFLOW)
        assert host.ctx.has(Keys.WORKSPACE)
        cfg = host.dump_config()
        assert cfg["plugins"] == host.dump()
        assert "executor" in cfg["services"]
        host.unload()
        assert host.dump() == []

    def test_dump_config_lists_each_default_profile(self, tmp_path: Path) -> None:
        chat = compose_chat(gateway=_FakeGateway(), scratch_dir=tmp_path / "chat")
        assert chat.dump_config()["plugins"] == ["run_stores", "tools", "llm"]
        assert "llm" in chat.dump_config()["services"]
        assert "agent_call" not in chat.dump_config()["services"]
        chat.unload()
        run = compose_run(run_id="r", run_dir=tmp_path / "run")
        assert run.dump_config()["plugins"] == [
            "run_stores",
            "executor",
            "workspace",
            "workflow",
        ]
        run.unload()

    def test_tool_belt_unregisters_on_unload(self, tmp_path: Path) -> None:
        from molexp.harness.host.plugins.tools import ToolBelt

        host = compose_plan(run_id="abcd1234", run_dir=tmp_path, gateway=_FakeGateway())
        belt = host.ctx.require(Keys.TOOLS)
        assert isinstance(belt, ToolBelt)
        belt.register(object(), host.ctx)
        assert len(belt.snapshot()) == 1
        host.unload()
        assert belt.snapshot() == ()


class TestToolBeltExecute:
    async def test_pre_execute_skip_next_skips_body(self) -> None:
        from molexp.harness.host.plugins.tools import ToolBelt

        ctx = Context()
        belt = ToolBelt()
        belt.bind(ctx)
        ran = {"n": 0}

        async def probe() -> str:
            ran["n"] += 1
            return "ran"

        probe.__name__ = "probe"
        belt.register(probe, ctx)

        async def block(payload: object, nxt: object) -> object:
            del nxt
            return {"result": "blocked", "payload": payload}

        ctx.on("tools/pre-execute", block, mode="waterfall")
        result = await belt.execute("probe", {})
        assert result == {"result": "blocked", "payload": {"name": "probe", "args": {}}}
        assert ran["n"] == 0

    async def test_execute_runs_body_when_pre_delegates(self) -> None:
        from molexp.harness.host.plugins.tools import ToolBelt

        ctx = Context()
        belt = ToolBelt()
        belt.bind(ctx)

        async def probe(*, x: int) -> int:
            return x + 1

        probe.__name__ = "probe"
        belt.register(probe, ctx)

        async def pass_through(payload: object, nxt: object) -> object:
            return await nxt(payload)  # type: ignore[misc,operator]

        ctx.on("tools/pre-execute", pass_through, mode="waterfall")
        assert await belt.execute("probe", {"x": 1}) == 2

    async def test_snapshot_wraps_body_through_execute(self) -> None:
        from molexp.harness.host.plugins.tools import ToolBelt

        ctx = Context()
        belt = ToolBelt()
        belt.bind(ctx)
        ran = {"n": 0}

        async def probe() -> str:
            ran["n"] += 1
            return "ran"

        probe.__name__ = "probe"
        belt.register(probe, ctx)

        async def block(payload: object, nxt: object) -> str:
            del payload, nxt
            return "wrapped"

        ctx.on("tools/pre-execute", block, mode="waterfall")
        wrapped = belt.snapshot()[0]
        assert await wrapped() == "wrapped"
        assert ran["n"] == 0


class _FakeApprovalStore:
    def __init__(self) -> None:
        self.grants: dict[str, object] = {}
        self.pending_calls: list[tuple[str, object]] = []
        self.decisions: list[object] = []

    def granted_decision_for(self, request_id: str) -> object | None:
        return self.grants.get(request_id)

    def record_pending(self, run_id: str, request: object) -> None:
        self.pending_calls.append((run_id, request))

    def record_decision(self, decision: object) -> None:
        self.decisions.append(decision)

    def pending(self, run_id: str) -> list[object]:
        del run_id
        return []


def _approval_request(request_id: str = "req-1") -> object:
    from datetime import UTC, datetime

    from molexp.harness.schemas import ApprovalRequest

    return ApprovalRequest(
        id=request_id,
        intent="overwrite",
        reason="test",
        triggered_by_policy="side_effects_present",
        metadata={},
        created_at=datetime.now(tz=UTC),
    )


class TestApprovalRequest:
    async def test_stored_grant_skips_later_listener(self) -> None:
        from datetime import UTC, datetime

        from molexp.harness.host.plugins.approval import ApprovalPlugin
        from molexp.harness.schemas import ApprovalDecision

        ctx = Context()
        store = _FakeApprovalStore()
        grant = ApprovalDecision(
            request_id="req-1",
            granted=True,
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
        )
        store.grants["req-1"] = grant
        ctx.provide(Keys.APPROVAL, store)
        ctx.provide(Keys.RUN_ID, "run-1")
        Host_ = Host
        host = Host_()
        host.ctx = ctx
        host.mount(ApprovalPlugin())
        ran = {"n": 0}

        async def late(value: object, nxt: object) -> object:
            ran["n"] += 1
            return await nxt(value)  # type: ignore[misc,operator]

        ctx.on("approval/request", late, mode="waterfall")
        result = await ctx.waterfall("approval/request", _approval_request())
        assert result is grant
        assert ran["n"] == 0

    async def test_missing_decision_records_pending(self) -> None:
        from molexp.harness.errors import ApprovalPendingError
        from molexp.harness.host.plugins.approval import ApprovalPlugin

        ctx = Context()
        store = _FakeApprovalStore()
        ctx.provide(Keys.APPROVAL, store)
        ctx.provide(Keys.RUN_ID, "run-1")
        host = Host()
        host.ctx = ctx
        host.mount(ApprovalPlugin())
        with pytest.raises(ApprovalPendingError):
            await ctx.waterfall("approval/request", _approval_request())
        assert len(store.pending_calls) == 1


class TestToolApproval:
    async def test_side_effect_tool_pending_skips_body(self) -> None:
        from molexp.harness.errors import ApprovalPendingError
        from molexp.harness.host.plugins.approval import ApprovalPlugin
        from molexp.harness.host.plugins.tools import ToolBelt

        ctx = Context()
        store = _FakeApprovalStore()
        ctx.provide(Keys.APPROVAL, store)
        ctx.provide(Keys.RUN_ID, "run-1")
        belt = ToolBelt()
        belt.bind(ctx)
        ctx.provide(Keys.TOOLS, belt)
        host = Host()
        host.ctx = ctx
        host.mount(ApprovalPlugin())
        ran = {"n": 0}

        async def boom() -> str:
            ran["n"] += 1
            return "nope"

        boom.__name__ = "boom"
        boom.side_effects = ["overwrite"]  # type: ignore[attr-defined]
        belt.register(boom, ctx)
        with pytest.raises(ApprovalPendingError):
            await belt.execute("boom", {})
        assert ran["n"] == 0

    async def test_read_only_tool_skips_approval(self) -> None:
        from molexp.harness.host.plugins.approval import ApprovalPlugin
        from molexp.harness.host.plugins.tools import ToolBelt

        ctx = Context()
        store = _FakeApprovalStore()
        ctx.provide(Keys.APPROVAL, store)
        ctx.provide(Keys.RUN_ID, "run-1")
        belt = ToolBelt()
        belt.bind(ctx)
        ctx.provide(Keys.TOOLS, belt)
        host = Host()
        host.ctx = ctx
        host.mount(ApprovalPlugin())

        async def ping() -> str:
            return "ok"

        ping.__name__ = "ping"
        belt.register(ping, ctx)
        assert await belt.execute("ping", {}) == "ok"
        assert store.pending_calls == []


class TestDomainPlugins:
    def test_compose_run_publishes_workspace_and_workflow(self, tmp_path: Path) -> None:
        host = compose_run(run_id="abcd1234", run_dir=tmp_path)
        assert host.ctx.workspace is host.ctx.require(Keys.WORKSPACE)
        assert host.ctx.workflow is host.ctx.require(Keys.WORKFLOW)
        host.unload()
        assert hasattr(host.ctx, "workspace") is False
        assert hasattr(host.ctx, "workflow") is False

    def test_compose_does_not_import_workspace_class(self) -> None:
        import inspect

        from molexp.harness.host import compose as compose_mod

        src = inspect.getsource(compose_mod)
        assert "from molexp.workspace.workspace import Workspace" not in src
        assert "WorkspacePlugin" in src
