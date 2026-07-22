"""RED tests for the side-effecting plan tools + the call-time dispatch entry
(plan-emergent-03, ac-005..007).

Pins two things that do NOT exist yet:

- ``molexp.harness.stages.invoke_capability.dispatch_capability`` — the reusable
  module-level coroutine extracted from ``InvokeCapability.run`` (validate →
  resolve → side-effect gate → materialize runner → execute → promote to a
  ``capability_invocation_result`` ``PlanArtifactRef``). ``InvokeCapability.run``
  now delegates to it (behavior-unchanged is guarded by the untouched
  ``test_invoke_capability_stage.py``).
- ``molexp.harness.plan_tools.run_capability`` / ``run_acceptance_test`` — thin
  agent-facing plan tools that route through ``dispatch_capability`` and never
  re-implement subprocess/runner materialization.

Function-level only: no ``LocalExecutor`` subprocess is spawned — the read-only
path uses ``DryRunExecutor`` (dispatch still resolves + persists a result in
process) and the rejection path is aborted by the gate BEFORE the executor runs.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest
from molexp.harness.plan_tools import PlanToolResult, run_acceptance_test, run_capability

from molexp.harness import CapabilityRegistry, DryRunExecutor, StageExecutionError
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.policy import enforce_side_effect_approvals
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.schemas import PlanArtifactRef, ToolCapability
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.stages import Approver, InvokeCapability, auto_grant_approver
from molexp.harness.stages.invoke_capability import dispatch_capability
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

_ECHO_PATH = "tests.test_harness._capability_fixtures:echo"

# ──────────────────────────────────────────────────────────── fixtures / helpers


def _make_capability(
    *,
    id_: str = "cap.invoke",
    callable_path: str | None = None,
    side_effects: list[str] | None = None,
) -> ToolCapability:
    """Build a ``ToolCapability`` with an explicit ``callable_path``."""
    return ToolCapability(
        id=id_,
        package="tests",
        name="Invoke",
        description="An invokable capability",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={"type": "object", "properties": {}},
        callable_path=callable_path,
        side_effects=side_effects or [],
    )


def _make_ctx(
    root: Path,
    *,
    registry: CapabilityRegistry | None = None,
) -> HarnessRunContext:
    """Build a ``HarnessRunContext`` backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-plan-tools-cap",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
        capability_registry=registry,
    )


def _fake_ref(id_: str = "artfake0001beef00") -> PlanArtifactRef:
    return PlanArtifactRef(
        id=id_,
        kind="capability_invocation_result",
        uri="file:///tmp/x.json",
        sha256="a" * 64,
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="test",
    )


async def denying_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Reject every request — proves the gate aborts before dispatch."""
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="test",
        decided_at=datetime.now(tz=UTC),
        reason="denied",
    )


_DENYING: Approver = denying_approver


# ──────────────────────────────────────────────────────────────── ac-005


class TestDispatchCapabilityEntry:
    """``dispatch_capability`` is a reusable async entry that ``run`` delegates to."""

    def test_dispatch_capability_is_an_async_callable(self) -> None:
        assert inspect.iscoroutinefunction(dispatch_capability)

    async def test_invoke_capability_run_delegates_to_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        fake = _fake_ref()

        async def _spy(
            ctx: HarnessRunContext,
            capability_id: str,
            parameters: dict[str, object],
            *,
            executor: object,
            approve: object,
            timeout_s: int,
        ) -> PlanArtifactRef:
            captured["capability_id"] = capability_id
            captured["parameters"] = parameters
            captured["executor"] = executor
            captured["approve"] = approve
            captured["timeout_s"] = timeout_s
            return fake

        monkeypatch.setattr("molexp.harness.stages.invoke_capability.dispatch_capability", _spy)
        ctx = _make_ctx(tmp_path)
        executor = DryRunExecutor()
        stage = InvokeCapability(
            "cap.echo",
            {"message": "hi"},
            executor=executor,
            approve=auto_grant_approver,
            timeout_s=123,
        )

        result = await stage.run(ctx)

        assert result is fake
        assert captured["capability_id"] == "cap.echo"
        assert captured["parameters"] == {"message": "hi"}
        assert captured["executor"] is executor
        assert captured["approve"] is auto_grant_approver
        assert captured["timeout_s"] == 123


# ──────────────────────────────────────────────────────────────── ac-006


class TestCapabilityToolsRouteThroughDispatch:
    """``run_capability`` / ``run_acceptance_test`` delegate to ``dispatch_capability``."""

    async def test_run_capability_calls_dispatch_and_wraps_the_ref(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        fake = _fake_ref()

        async def _spy(
            ctx: HarnessRunContext,
            capability_id: str,
            parameters: dict[str, object],
            *,
            executor: object,
            approve: object,
            timeout_s: int,
        ) -> PlanArtifactRef:
            captured["capability_id"] = capability_id
            captured["parameters"] = parameters
            captured["executor"] = executor
            return fake

        monkeypatch.setattr("molexp.harness.plan_tools.capability_tools.dispatch_capability", _spy)
        ctx = _make_ctx(tmp_path)
        executor = DryRunExecutor()

        result = await run_capability(
            ctx=ctx,
            capability_id="cap.echo",
            parameters={"message": "hi"},
            executor=executor,
            approve=auto_grant_approver,
        )

        assert captured["capability_id"] == "cap.echo"
        assert captured["parameters"] == {"message": "hi"}
        assert captured["executor"] is executor
        assert isinstance(result, PlanToolResult)
        assert result.ok is True
        assert result.artifact_ref == fake.id

    async def test_run_acceptance_test_calls_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        fake = _fake_ref()

        async def _spy(
            ctx: HarnessRunContext,
            capability_id: str,
            parameters: dict[str, object],
            *,
            executor: object,
            approve: object,
            timeout_s: int,
        ) -> PlanArtifactRef:
            captured["capability_id"] = capability_id
            captured["parameters"] = parameters
            return fake

        monkeypatch.setattr("molexp.harness.plan_tools.capability_tools.dispatch_capability", _spy)
        ctx = _make_ctx(tmp_path)

        result = await run_acceptance_test(
            ctx=ctx,
            parameters={"suite": "unit"},
            executor=DryRunExecutor(),
            approve=auto_grant_approver,
        )

        # The fixed acceptance-test capability id is dispatched (value is the
        # tool's own concern; we only pin that it routed through dispatch).
        assert "capability_id" in captured
        assert captured["parameters"] == {"suite": "unit"}
        assert isinstance(result, PlanToolResult)
        assert result.artifact_ref == fake.id

    async def test_run_capability_real_dispatch_persists_invocation_result(
        self, tmp_path: Path
    ) -> None:
        cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH, side_effects=[])
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))

        result = await run_capability(
            ctx=ctx,
            capability_id="cap.echo",
            parameters={"message": "hi"},
            executor=DryRunExecutor(),
            approve=auto_grant_approver,
        )

        assert isinstance(result, PlanToolResult)
        assert result.ok is True
        persisted = ctx.artifact_store.latest_by_kind("capability_invocation_result")
        assert persisted is not None
        assert result.artifact_ref == persisted.id

    def test_capability_tools_does_not_reimplement_the_runner_path(self) -> None:
        """``capability_tools.py`` must route through dispatch, not re-materialize."""
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "molexp"
            / "harness"
            / "plan_tools"
            / "capability_tools.py"
        ).read_text(encoding="utf-8")

        for forbidden in ("_RUNNER_SOURCE", "mkstemp", "_materialize_runner"):
            assert forbidden not in source, (
                f"capability_tools.py must route through dispatch_capability, "
                f"not re-implement {forbidden!r}"
            )


# ──────────────────────────────────────────────────────────────── ac-007


class TestSideEffectGateFiresOnceBeforeDispatch:
    """A rejected destructive capability aborts pre-dispatch with a single gate."""

    async def test_rejected_destructive_capability_aborts_before_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = _make_capability(id_="cap.del", callable_path=_ECHO_PATH, side_effects=["delete"])
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
        counter = {"n": 0}
        real_enforce = enforce_side_effect_approvals

        async def _counting(*args: object, **kwargs: object) -> object:
            counter["n"] += 1
            return await real_enforce(*args, **kwargs)  # type: ignore

        monkeypatch.setattr(
            "molexp.harness.stages.invoke_capability.enforce_side_effect_approvals",
            _counting,
        )

        with pytest.raises(StageExecutionError):
            await run_capability(
                ctx=ctx,
                capability_id="cap.del",
                parameters={"message": "hi"},
                executor=DryRunExecutor(),
                approve=_DENYING,
            )

        # Entered exactly once (no double gate) and never dispatched.
        assert counter["n"] == 1
        assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []
