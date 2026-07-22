"""Regression: the process-internal plan-tool surface, offline.

Binding runtime example for spec ``plan-emergent-03-plan-tools`` (ac-010). A
library-external user drives the **public** plan-tool surface
(``molexp.harness.plan_tools``) end to end — the board-state tools, the
side-effecting capability tools, and the harness->agent-loop adapter — using
only the documented import surfaces. No network, no subprocess (the read-only
capability path runs through :class:`DryRunExecutor`, which never spawns one),
no server, no CLI.

Four ac-010 reference scenarios:

1. **board transition** — a fake :class:`TaskBoardHandle` carrying task ``t1``
   (status ``pending``); ``update_task`` walks it to ``in_progress`` and
   ``complete_task`` to ``done``. Each board immutable-write returns a NEW board
   and the transitions land (``pending -> in_progress -> done``) while every
   input board is left byte-identical (immutability law).
2. **read-only capability** — a ``ToolCapability`` with ``side_effects == []``
   dispatched through ``run_capability`` (and ``run_acceptance_test``) with a
   :class:`DryRunExecutor` + ``auto_grant_approver`` returns
   ``PlanToolResult.ok is True`` and persists a ``capability_invocation_result``.
3. **destructive capability rejected pre-dispatch** — a ``ToolCapability`` with a
   non-empty ``side_effects`` list dispatched under a REJECTING approver raises
   :class:`StageExecutionError` BEFORE dispatch, persisting nothing.
4. **adapter** — ``as_loop_tool`` on a read-only board tool writes ZERO approval
   artifacts (gate bypass); on a side-effect tool with a granting approver the
   side-effect gate fires exactly once.

Deterministic + offline. Run standalone with
``python regressions/plan-emergent-03-plan-tools.py``; the final line on success
is ``plan-emergent-03-plan-tools: ok``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from molexp.harness import (
    DryRunExecutor,
    FileArtifactStore,
    HarnessRunContext,
    SQLiteArtifactLineageStore,
    SQLiteEventLog,
    StageExecutionError,
)
from molexp.harness.plan_tools import (
    BOARD_TOOLS,
    PlanTool,
    PlanToolResult,
    TaskBoardHandle,
    as_loop_tool,
    block_task,
    complete_task,
    inspect_task,
    list_tasks,
    run_acceptance_test,
    run_capability,
    update_task,
)
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.schemas import ToolCapability
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.stages import Approver, auto_grant_approver

# The fixed capability id ``run_acceptance_test`` always dispatches. It is part of
# that tool's documented contract, registered here under the same id.
_ACCEPTANCE_TEST_CAPABILITY_ID = "molexp.acceptance_test"


# ─────────────────────────────────────────────────────────── fake board + ctx


@dataclasses.dataclass(frozen=True)
class _FakeTask:
    """A minimal frozen task record for the fake board."""

    id: str
    name: str
    status: str


class _FakeBoard:
    """In-memory board satisfying the ``TaskBoardHandle`` Protocol.

    Read methods return live data; every immutable-write method returns a NEW
    ``_FakeBoard`` with the transition applied and appends it to ``produced``
    WITHOUT mutating ``self`` — so a tool that routes a mutation through the
    handle leaves the input board byte-identical while the produced board is
    observable (the plan tools discard the returned board, so ``produced`` is
    how a caller reaches it).
    """

    def __init__(self, tasks: list[_FakeTask]) -> None:
        self._tasks: tuple[_FakeTask, ...] = tuple(tasks)
        self.produced: list[_FakeBoard] = []

    def list_tasks(self) -> list[_FakeTask]:
        return list(self._tasks)

    def get_task(self, task_id: str) -> _FakeTask:
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def get_artifact(self, artifact_id: str) -> dict[str, object]:
        raise KeyError(artifact_id)

    def _with_status(self, task_id: str, status: str) -> _FakeBoard:
        updated = [
            dataclasses.replace(task, status=status) if task.id == task_id else task
            for task in self._tasks
        ]
        child = _FakeBoard(updated)
        self.produced.append(child)
        return child

    def with_task_updated(self, task_id: str, changes: dict[str, object]) -> _FakeBoard:
        status = changes.get("status")
        new_status = status if isinstance(status, str) else self.get_task(task_id).status
        return self._with_status(task_id, new_status)

    def mark_complete(self, task_id: str) -> _FakeBoard:
        return self._with_status(task_id, "done")

    def mark_blocked(self, task_id: str, reason: str) -> _FakeBoard:
        del reason
        return self._with_status(task_id, "blocked")

    def apply_patch(self, patch: dict[str, object]) -> _FakeBoard:
        del patch
        child = _FakeBoard(list(self._tasks))
        self.produced.append(child)
        return child


def _status(board: _FakeBoard, task_id: str) -> str:
    return board.get_task(task_id).status


def _make_ctx(
    root: Path,
    *,
    registry: InMemoryCapabilityRegistry | None = None,
) -> HarnessRunContext:
    """Build a ``HarnessRunContext`` backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-plan-emergent-03",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
        capability_registry=registry,
    )


def _capability(
    *,
    id_: str,
    side_effects: list[str],
    properties: dict[str, object],
    required: list[str],
) -> ToolCapability:
    """A ``ToolCapability`` whose ``callable_path`` resolves but is never run.

    ``dispatch_capability`` resolves ``callable_path`` in-process as a fail-fast
    guard; under :class:`DryRunExecutor` the callable itself is never invoked, so
    a stdlib callable (``builtins:dict``) is a resolvable, offline no-op target.
    """
    return ToolCapability(
        id=id_,
        package="regressions",
        name="Invoke",
        description="A regression capability",
        input_schema={"type": "object", "properties": properties, "required": required},
        output_schema={"type": "object", "properties": {}},
        callable_path="builtins:dict",
        side_effects=side_effects,
    )


async def _denying_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Reject every request — proves the side-effect gate aborts pre-dispatch."""
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="regression",
        decided_at=datetime.now(tz=UTC),
        reason="denied",
    )


_DENYING: Approver = _denying_approver


# ─────────────────────────────────────────────────────────── ac-010 scenario 1


async def _scenario_board_transition() -> None:
    """A board walks ``pending -> in_progress -> done`` via the board tools."""
    with TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        board0 = _FakeBoard(
            [
                _FakeTask(id="t1", name="Build", status="pending"),
                _FakeTask(id="t2", name="Test", status="pending"),
            ]
        )
        assert isinstance(board0, TaskBoardHandle), "fake board must satisfy TaskBoardHandle"

        listed = await list_tasks(ctx=ctx, board=board0)
        inspected = await inspect_task(ctx=ctx, board=board0, task_id="t1")

        # pending -> in_progress -> done, then block the sibling task.
        r_update = await update_task(
            ctx=ctx, board=board0, task_id="t1", changes={"status": "in_progress"}
        )
        board1 = board0.produced[-1]
        r_complete = await complete_task(ctx=ctx, board=board1, task_id="t1")
        board2 = board1.produced[-1]
        r_block = await block_task(ctx=ctx, board=board2, task_id="t2", reason="waiting on t1")
        board3 = board2.produced[-1]

        # Reference observations (printed before asserting).
        print(f"board: initial t1 status        = {_status(board0, 't1')!r}")
        print(f"board: list_tasks count         = {listed.data.get('count')}")
        print(f"board: inspect_task ok          = {inspected.ok}")
        print(f"board: after update_task t1     = {_status(board1, 't1')!r}")
        print(f"board: after complete_task t1   = {_status(board2, 't1')!r}")
        print(f"board: after block_task t2      = {_status(board3, 't2')!r}")
        print(f"board: input t1 unchanged       = {_status(board0, 't1')!r}")

        # Each immutable-write returns a NEW board object.
        assert board1 is not board0, "update_task must route through a NEW board"
        assert board2 is not board1, "complete_task must route through a NEW board"
        assert board3 is not board2, "block_task must route through a NEW board"

        # The transition chain lands on the produced boards.
        assert _status(board1, "t1") == "in_progress", "update_task did not land in_progress"
        assert _status(board2, "t1") == "done", "complete_task did not land done"
        assert _status(board3, "t2") == "blocked", "block_task did not land blocked"

        # Every input board is left byte-identical (immutability law).
        assert _status(board0, "t1") == "pending", "input board0 was mutated"
        assert _status(board1, "t1") == "in_progress", "input board1 was mutated"
        assert _status(board2, "t2") == "pending", "input board2 was mutated"

        # Every tool returns the uniform PlanToolResult payload.
        for result in (listed, inspected, r_update, r_complete, r_block):
            assert isinstance(result, PlanToolResult), "board tools return PlanToolResult"
            assert result.ok is True, "board tool reported failure"


# ─────────────────────────────────────────────────────────── ac-010 scenario 2


async def _scenario_read_only_capability() -> None:
    """A read-only capability dispatches ok through DryRunExecutor + auto-grant."""
    with TemporaryDirectory() as tmp:
        registry = InMemoryCapabilityRegistry(
            [
                _capability(
                    id_="cap.echo",
                    side_effects=[],
                    properties={"message": {"type": "string"}},
                    required=["message"],
                ),
                _capability(
                    id_=_ACCEPTANCE_TEST_CAPABILITY_ID,
                    side_effects=[],
                    properties={"suite": {"type": "string"}},
                    required=[],
                ),
            ]
        )
        ctx = _make_ctx(Path(tmp), registry=registry)

        result = await run_capability(
            ctx=ctx,
            capability_id="cap.echo",
            parameters={"message": "hi"},
            executor=DryRunExecutor(),
            approve=auto_grant_approver,
        )
        # Capture the persisted result BEFORE the second dispatch so latest_by_kind
        # still names the cap.echo invocation.
        persisted = ctx.artifact_store.latest_by_kind("capability_invocation_result")
        acceptance = await run_acceptance_test(
            ctx=ctx,
            parameters={"suite": "unit"},
            executor=DryRunExecutor(),
            approve=auto_grant_approver,
        )
        all_ids = {
            ref.id for ref in ctx.artifact_store.list_by_kind("capability_invocation_result")
        }

        # Reference observations (printed before asserting).
        print(f"read-only cap: run_capability ok       = {result.ok}")
        print(f"read-only cap: artifact_ref            = {result.artifact_ref!r}")
        print(f"read-only cap: run_acceptance_test ok  = {acceptance.ok}")
        print(
            f"read-only cap: persisted result id     = {None if persisted is None else persisted.id!r}"
        )
        print(f"read-only cap: persisted result count  = {len(all_ids)}")

        assert isinstance(result, PlanToolResult), "run_capability returns PlanToolResult"
        assert result.ok is True, "read-only capability must dispatch ok"
        assert result.artifact_ref is not None, "a dispatched result must carry an artifact_ref"
        assert persisted is not None, "the capability_invocation_result must be persisted"
        assert result.artifact_ref == persisted.id, (
            "artifact_ref must point at the persisted result"
        )
        assert acceptance.ok is True, "run_acceptance_test must dispatch ok"
        assert acceptance.artifact_ref in all_ids, (
            "run_acceptance_test result must be persisted too"
        )


# ─────────────────────────────────────────────────────────── ac-010 scenario 3


async def _scenario_destructive_rejected() -> None:
    """A destructive capability under a rejecting approver aborts pre-dispatch."""
    with TemporaryDirectory() as tmp:
        registry = InMemoryCapabilityRegistry(
            [
                _capability(
                    id_="cap.del",
                    side_effects=["delete"],
                    properties={"message": {"type": "string"}},
                    required=["message"],
                )
            ]
        )
        ctx = _make_ctx(Path(tmp), registry=registry)

        raised = False
        try:
            await run_capability(
                ctx=ctx,
                capability_id="cap.del",
                parameters={"message": "hi"},
                executor=DryRunExecutor(),
                approve=_DENYING,
            )
        except StageExecutionError:
            raised = True
        persisted = ctx.artifact_store.list_by_kind("capability_invocation_result")

        # Reference observations (printed before asserting).
        print(f"destructive cap: raised StageExecutionError = {raised}")
        print(f"destructive cap: results persisted          = {len(persisted)}")

        assert raised, "a denied destructive capability must raise StageExecutionError"
        assert persisted == [], "a rejected gate must abort BEFORE any dispatch/persistence"


# ─────────────────────────────────────────────────────────── ac-010 scenario 4


async def _scenario_adapter() -> None:
    """``as_loop_tool`` bypasses the gate read-only and fires it once otherwise."""
    # 4a. read-only board tool: denying approver never consulted, no artifact.
    with TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        board = _FakeBoard([_FakeTask(id="t1", name="Build", status="pending")])
        list_tool = next(tool for tool in BOARD_TOOLS if tool.name == "list_tasks")
        adapted_ro = as_loop_tool(list_tool, ctx=ctx, approve=_DENYING)

        result = await adapted_ro(ctx=ctx, board=board)
        approvals = ctx.artifact_store.list_by_kind("side_effect_approval")

        print(f"adapter read-only: result ok          = {result.ok}")
        print(f"adapter read-only: approval artifacts = {len(approvals)}")

        assert isinstance(result, PlanToolResult), "adapted read-only tool returns PlanToolResult"
        assert result.ok is True, "read-only tool body must run"
        assert approvals == [], "a read-only tool must write ZERO approval artifacts (gate bypass)"

    # 4b. side-effect tool: granting approver fires the gate exactly once.
    with TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))

        async def _delete_body(**kwargs: object) -> PlanToolResult:
            del kwargs
            return PlanToolResult(ok=True, summary="deleted", data={}, artifact_ref=None)

        delete_tool = PlanTool(
            name="run_delete",
            description="Delete something.",
            fn=_delete_body,
            input_schema={"type": "object"},
            side_effects=["delete"],
        )
        adapted_se = as_loop_tool(delete_tool, ctx=ctx, approve=auto_grant_approver)

        result = await adapted_se()
        approvals = ctx.artifact_store.list_by_kind("side_effect_approval")

        print(f"adapter side-effect: result ok          = {result.ok}")
        print(f"adapter side-effect: approval artifacts = {len(approvals)}")

        assert result.ok is True, "granted side-effect tool body must run"
        assert len(approvals) == 1, "the side-effect gate must fire EXACTLY once"


async def main() -> int:
    """Run all four ac-010 scenarios; print the success marker as the final line."""
    await _scenario_board_transition()
    await _scenario_read_only_capability()
    await _scenario_destructive_rejected()
    await _scenario_adapter()
    print("plan-emergent-03-plan-tools: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
