"""RED tests for the plan-tool board-state facade (plan-emergent-03, ac-001..004).

Pins the process-internal board tool surface that does NOT exist yet
(``molexp.harness.plan_tools``): the :class:`PlanTool` descriptor, the
:class:`PlanToolResult` payload, the :class:`TaskBoardHandle` Protocol, and the
eight board-state tools (``list_tasks`` / ``inspect_task`` / ``inspect_artifact``
read-only; ``update_task`` / ``complete_task`` / ``block_task`` /
``propose_plan_patch`` mutating via the handle's immutable writes).

Function-level only: every tool is driven directly with a fake board satisfying
``TaskBoardHandle`` and a real :class:`HarnessRunContext` (needed by
``inspect_artifact``, which reads through ``ctx.artifact_store``). No subprocess,
no server, no CLI. ``asyncio_mode = "auto"`` (pyproject) runs the ``async def``
tests without an explicit marker; the descriptor/side-effect tests are plain sync.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import ArtifactNotFoundError, HarnessError
from molexp.harness.plan_tools import (
    BOARD_TOOLS,
    PlanTool,
    PlanToolResult,
    TaskBoardHandle,
    block_task,
    complete_task,
    inspect_artifact,
    inspect_task,
    list_tasks,
    propose_plan_patch,
    update_task,
)
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

# ──────────────────────────────────────────────────────────── fixtures / helpers

_BoardCall = tuple[str, tuple[object, ...], dict[str, object]]


@dataclasses.dataclass(frozen=True)
class _FakeTask:
    """A minimal frozen task record for the fake board."""

    id: str
    name: str
    status: str


class _FakeBoard:
    """In-memory stand-in satisfying the ``TaskBoardHandle`` Protocol.

    Read methods return live data; every immutable-write method appends a
    ``(name, args, kwargs)`` record to a shared ``calls`` list and returns a
    fresh ``_FakeBoard`` WITHOUT mutating ``self`` — so a tool that routes
    through the handle both leaves the input board byte-identical and is
    observable through ``calls``.
    """

    def __init__(self, tasks: list[_FakeTask], *, calls: list[_BoardCall] | None = None) -> None:
        self._tasks: tuple[_FakeTask, ...] = tuple(tasks)
        self.calls: list[_BoardCall] = calls if calls is not None else []

    def list_tasks(self) -> list[_FakeTask]:
        return list(self._tasks)

    def get_task(self, task_id: str) -> _FakeTask:
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def get_artifact(self, artifact_id: str) -> dict[str, object]:
        raise KeyError(artifact_id)

    def place(self, *args: object, **kwargs: object) -> _FakeBoard:
        self.calls.append(("place", args, kwargs))
        return _FakeBoard(list(self._tasks), calls=self.calls)

    def with_task_updated(self, *args: object, **kwargs: object) -> _FakeBoard:
        self.calls.append(("with_task_updated", args, kwargs))
        return _FakeBoard(list(self._tasks), calls=self.calls)

    def mark_complete(self, *args: object, **kwargs: object) -> _FakeBoard:
        self.calls.append(("mark_complete", args, kwargs))
        return _FakeBoard(list(self._tasks), calls=self.calls)

    def mark_blocked(self, *args: object, **kwargs: object) -> _FakeBoard:
        self.calls.append(("mark_blocked", args, kwargs))
        return _FakeBoard(list(self._tasks), calls=self.calls)

    def apply_patch(self, *args: object, **kwargs: object) -> _FakeBoard:
        self.calls.append(("apply_patch", args, kwargs))
        return _FakeBoard(list(self._tasks), calls=self.calls)


def _make_board() -> _FakeBoard:
    return _FakeBoard(
        [
            _FakeTask(id="t1", name="Build", status="pending"),
            _FakeTask(id="t2", name="Test", status="pending"),
        ]
    )


def _snapshot(board: _FakeBoard) -> list[tuple[str, str]]:
    return [(task.id, task.status) for task in board.list_tasks()]


def _make_ctx(root: Path) -> HarnessRunContext:
    """Build a ``HarnessRunContext`` backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-plan-tools-board",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


async def _noop_fn(**kwargs: object) -> PlanToolResult:
    return PlanToolResult(ok=True, summary="", data={}, artifact_ref=None)


def _make_plan_tool() -> PlanTool:
    return PlanTool(
        name="probe",
        description="A probe tool.",
        input_schema={"type": "object"},
        side_effects=[],
        fn=_noop_fn,
    )


# The seven board-state tool functions + the immutable-write handle method each
# routes through. ``inspect_artifact`` takes the uniform ``(*, ctx, board, ...)``
# signature even though it reads through ``ctx.artifact_store``, not the board.
_WRITE_TOOL_CASES = [
    pytest.param(
        update_task,
        {"task_id": "t1", "changes": {"name": "renamed"}},
        "with_task_updated",
        id="update_task",
    ),
    pytest.param(complete_task, {"task_id": "t1"}, "mark_complete", id="complete_task"),
    pytest.param(
        block_task, {"task_id": "t1", "reason": "waiting on t0"}, "mark_blocked", id="block_task"
    ),
    pytest.param(
        propose_plan_patch, {"patch": {"op": "noop"}}, "apply_patch", id="propose_plan_patch"
    ),
]


# ──────────────────────────────────────────────────────────────── ac-001


class TestPlanToolDescriptor:
    """``PlanTool`` is a frozen dataclass; ``PlanToolResult`` a frozen pydantic model."""

    def test_plan_tool_exposes_declared_fields(self) -> None:
        tool = _make_plan_tool()

        assert tool.name == "probe"
        assert tool.description == "A probe tool."
        assert tool.input_schema == {"type": "object"}
        assert tool.side_effects == []
        assert tool.fn is _noop_fn

    def test_plan_tool_is_a_frozen_dataclass(self) -> None:
        tool = _make_plan_tool()

        with pytest.raises(dataclasses.FrozenInstanceError):
            tool.name = "changed"  # ty: ignore[invalid-assignment]

    def test_plan_tool_result_exposes_declared_fields(self) -> None:
        result = PlanToolResult(ok=True, summary="did it", data={"n": 1}, artifact_ref="art-1")

        assert result.ok is True
        assert result.summary == "did it"
        assert result.data == {"n": 1}
        assert result.artifact_ref == "art-1"

    def test_plan_tool_result_is_a_frozen_pydantic_model(self) -> None:
        result = PlanToolResult(ok=True, summary="", data={}, artifact_ref=None)

        with pytest.raises(ValidationError):
            result.ok = False  # ty: ignore[invalid-assignment]


class TestTaskBoardHandleProtocol:
    """``TaskBoardHandle`` is a runtime_checkable Protocol a fake board satisfies."""

    def test_fake_board_satisfies_the_protocol(self) -> None:
        assert isinstance(_make_board(), TaskBoardHandle)

    def test_bare_object_does_not_satisfy_the_protocol(self) -> None:
        assert not isinstance(object(), TaskBoardHandle)


# ──────────────────────────────────────────────────────────────── ac-002


class TestBoardToolsReadWrite:
    """Read tools never write; write tools route through immutable writes only."""

    async def test_list_tasks_is_read_only(self, tmp_path: Path) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)

        result = await list_tasks(ctx=ctx, board=board)

        assert isinstance(result, PlanToolResult)
        assert result.ok is True
        assert board.calls == []

    async def test_inspect_task_is_read_only(self, tmp_path: Path) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)

        result = await inspect_task(ctx=ctx, board=board, task_id="t1")

        assert isinstance(result, PlanToolResult)
        assert result.ok is True
        assert board.calls == []

    async def test_inspect_artifact_reads_through_artifact_store(self, tmp_path: Path) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)
        ref = ctx.artifact_store.put_json(
            kind="log", obj={"k": "v"}, created_by="test", parent_ids=[]
        )

        result = await inspect_artifact(ctx=ctx, board=board, artifact_id=ref.id)

        assert isinstance(result, PlanToolResult)
        assert result.ok is True
        assert board.calls == []

    @pytest.mark.parametrize(("tool_fn", "call_kwargs", "expected_method"), _WRITE_TOOL_CASES)
    async def test_write_tool_routes_through_immutable_write_and_never_mutates_input(
        self,
        tmp_path: Path,
        tool_fn: Callable[..., Awaitable[PlanToolResult]],
        call_kwargs: dict[str, object],
        expected_method: str,
    ) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)
        before = _snapshot(board)

        result = await tool_fn(ctx=ctx, board=board, **call_kwargs)

        assert isinstance(result, PlanToolResult)
        # The tool obtained a NEW board via exactly one immutable-write call.
        assert [name for name, _args, _kwargs in board.calls] == [expected_method]
        # The input board object is byte-identical — immutability preserved.
        assert _snapshot(board) == before


# ──────────────────────────────────────────────────────────────── ac-003


class TestBoardToolsDeclareEmptySideEffects:
    """All seven board tools present ``side_effects == []`` (gate bypass)."""

    def test_seven_board_tools_are_registered(self) -> None:
        assert len(BOARD_TOOLS) == 8

    def test_every_board_tool_declares_empty_side_effects(self) -> None:
        assert all(tool.side_effects == [] for tool in BOARD_TOOLS)

    def test_board_tool_names_cover_the_full_facade(self) -> None:
        assert {tool.name for tool in BOARD_TOOLS} == {
            "list_tasks",
            "place_task",
            "inspect_task",
            "inspect_artifact",
            "update_task",
            "complete_task",
            "block_task",
            "propose_plan_patch",
        }


# ──────────────────────────────────────────────────────────────── ac-004


class TestMissingIdRaisesTypedError:
    """Missing-id board reads raise a typed error — never a silent ``None``."""

    async def test_inspect_task_unknown_id_raises_typed_error(self, tmp_path: Path) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)

        with pytest.raises((KeyError, HarnessError)):
            await inspect_task(ctx=ctx, board=board, task_id="does-not-exist")

    async def test_inspect_artifact_unknown_id_raises_artifact_not_found(
        self, tmp_path: Path
    ) -> None:
        board = _make_board()
        ctx = _make_ctx(tmp_path)

        with pytest.raises(ArtifactNotFoundError):
            await inspect_artifact(ctx=ctx, board=board, artifact_id="no-such-artifact")
