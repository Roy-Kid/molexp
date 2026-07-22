"""RED tests for the production ``InteractiveLoopPlanRunner`` (plan-emergent-05c).

``InteractiveLoopPlanRunner`` (``molexp.harness.modes.emergent_plan``, not yet
written) is the private ``PlanLoopRunner`` impl and the **only** harness site
importing ``molexp.agent.loops`` (lazily). It builds a phase-02 ``InteractiveLoop``
+ ``AgentRuntime`` + sink from the 05a gateway's ``Router`` and drives the
emergent planning loop, injecting the phase-03 plan tools and the phase-05b
``EmergentPlanFormValidator``-backed ``should_stop`` guard.

These tests pin ac-005 offline with a **scripted fake ``Router``**: driving
``run_planning`` invokes ``Router.stream_agentic`` with the ``as_loop_tool``-
adapted plan tools and a non-None ``should_stop`` guard; and an
``EmergentPlanFormValidator``-backed guard denies a form-invalid board (steer)
while allowing a form-valid one to terminate.

BLOCKERS this test surfaces (see the file report):

1. **No public Router accessor.** ``RouterBackedAgentGateway`` (the 05a gateway)
   stores its router as ``self._router`` with **no** public accessor — the
   ``AgentGateway`` Protocol is only ``async call(spec)``. The runner needs a
   ``Router`` from the gateway. This test pins the required accessor as
   ``gateway.router``; 05a/05c MUST add it.
2. **No plan-tool injection seam on ``InteractiveLoop``.** ``InteractiveLoop.run``
   hard-codes its ops tools (``build_ops_tools``) and accepts no injected
   ``tools`` — so it cannot forward the plan tools to ``stream_agentic`` today.
   This test pins that the adapted plan tools reach ``stream_agentic``; a
   plan-tool injection seam on the loop is required for GREEN.

RED until ``molexp.harness.modes.emergent_plan`` exists (and the blockers land).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.loops import HookOutcome, LoopHooks, LoopState
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    McpToolSpec,
    ModelTier,
    RouterTextResult,
)
from molexp.agent.types import UsageBreakdown
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.plan import (
    BoardTask,
    Difficulty,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
)
from molexp.harness.plan_tools import BOARD_TOOLS, as_loop_tool
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.harness.validators import EmergentPlanFormValidator

# RED: the module + its private loop runner do not exist yet.
from molexp.harness.modes.emergent_plan import (  # noqa: E402 — intentional RED import
    InteractiveLoopPlanRunner,
)

pytestmark = pytest.mark.asyncio

_SPEC = {"title": "Zwitterion CG", "objective": "measure area-per-lipid"}


def _valid_board() -> TaskBoard:
    return TaskBoard(
        version=1,
        tasks=(
            BoardTask(
                id="t1",
                name="build system",
                acceptance=("energy is finite",),
                feasibility=FeasibilityAnnotation(reachable=True, difficulty=Difficulty.TRIVIAL),
            ),
        ),
    )


def _malformed_board() -> TaskBoard:
    return TaskBoard(version=1, tasks=(BoardTask(id="t1", name="build", acceptance=()),))


class _RecordingRouter:
    """Scripted fake ``Router`` recording every ``stream_agentic`` call.

    Yields a single ``FinalChunk`` so the loop terminates; records the ``tools``
    and ``should_stop`` it was handed so a test can assert the runner forwarded
    the adapted plan tools + form guard.
    """

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: Any = None,
        after_tool: Any = None,
        should_stop: Any = None,
    ) -> AsyncIterator[AgenticChunk]:
        del system, toolsets, tier, message_history, before_tool, after_tool
        self.stream_calls.append(
            {"prompt": prompt, "tools": tuple(tools), "should_stop": should_stop}
        )
        yield FinalChunk(text="planning done")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text="")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[Any],
        node_id: str = "",
        mcp_tools: tuple[McpToolSpec, ...] = (),
    ) -> Any:
        raise AssertionError("the planning loop should not call complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _FakeRouterGateway:
    """A fake ``AgentGateway`` that ALSO exposes a public ``Router`` accessor.

    BLOCKER: ``RouterBackedAgentGateway`` exposes no such accessor today (see the
    module docstring). This fake pins the accessor the loop runner must consume.
    """

    def __init__(self, router: _RecordingRouter) -> None:
        self.router = router

    async def call(self, spec: Any) -> Any:
        raise AssertionError("the loop runner drives the Router, never gateway.call")


def _ctx(tmp_path: Path, gateway: _FakeRouterGateway) -> HarnessRunContext:
    db = tmp_path / "harness.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    return HarnessRunContext(
        run_id="run-loop",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=artifacts),
        agent_gateway=gateway,  # type: ignore[arg-type]
    )


class _BoardHolder:
    def __init__(self, board: TaskBoard) -> None:
        self.board = board


def _form_guard(holder: _BoardHolder):
    """A ``ShouldStopGuard`` backed by ``EmergentPlanFormValidator`` over a board."""

    async def should_stop(*, state: LoopState) -> HookOutcome:
        del state
        report = EmergentPlanFormValidator.validate(
            ExperimentPlan(spec=_SPEC, board=holder.board)
        )
        if report.passed:
            return HookOutcome.proceed()
        return HookOutcome.deny("; ".join(v.message for v in report.violations))

    return should_stop


class TestFormValidatorShouldStopGuard:
    async def test_guard_denies_invalid_board_and_proceeds_on_valid(self) -> None:
        """ac-005: form-invalid ⇒ deny (steer); form-valid ⇒ proceed (terminate)."""
        holder = _BoardHolder(_malformed_board())
        guard = _form_guard(holder)

        assert (await guard(state=LoopState(step=1))).is_deny
        holder.board = _valid_board()
        assert (await guard(state=LoopState(step=2))).is_proceed


class TestInteractiveLoopPlanRunnerWiring:
    async def test_run_planning_streams_plan_tools_and_the_guard(self, tmp_path: Path) -> None:
        """ac-005: the runner drives stream_agentic with adapted plan tools + should_stop."""
        router = _RecordingRouter()
        ctx = _ctx(tmp_path, _FakeRouterGateway(router))
        tools = tuple(as_loop_tool(tool, ctx=ctx) for tool in BOARD_TOOLS)
        holder = _BoardHolder(_valid_board())
        hooks = LoopHooks(should_stop=_form_guard(holder))

        runner = InteractiveLoopPlanRunner()
        await runner.run_planning(
            ctx=ctx,
            board=holder.board,
            tools=tools,
            hooks=hooks,
            user_input="draft the plan",
        )

        assert router.stream_calls, "the runner must drive Router.stream_agentic"
        call = router.stream_calls[-1]
        assert call["tools"], "stream_agentic must receive the adapted plan tools"
        streamed_names = {getattr(t, "__name__", "") for t in call["tools"]}
        assert "list_tasks" in streamed_names  # as_loop_tool copies tool.name onto __name__
        assert call["should_stop"] is not None
