"""Tests for the side-effect approval gate (workspace-curation-toolset, link 03).

Contract under test (RED — the helpers + the ``InvokeCapability(approve=...)``
keyword do not exist yet, so this module fails at *import/collection* until the
production code lands):

- :func:`molexp.harness.make_side_effect_approval_requests` (pure) derives one
  :class:`~molexp.harness.schemas.approval.ApprovalRequest` per item carrying a
  NON-EMPTY ``side_effects`` list, reusing the existing ``"overwrite"``
  :data:`~molexp.harness.schemas.approval.ApprovalIntent` (no schema change),
  tagged ``triggered_by_policy == "side_effects_present"``, with deduped +
  sorted side effects in ``metadata``. Read-only items emit nothing.
- :func:`molexp.harness.enforce_side_effect_approvals` (async) is a true
  bypass when no item declares a side effect (returns ``None``, runs no gate,
  persists no artifact); otherwise it drives an :class:`ApprovalGate`
  (``result_kind="side_effect_approval"``) and lets a denial's
  :class:`StageExecutionError` propagate unwrapped.
- :class:`molexp.harness.InvokeCapability` gains a keyword-only ``approve``
  param that interposes the gate AFTER capability resolution and BEFORE any
  dispatch/persistence: a denied destructive capability never reaches the
  executor (no ``capability_invocation_result``), while a read-only capability
  bypasses the gate entirely and dispatches normally.

Async style: this module mirrors ``test_invoke_capability_stage.py`` and
``test_phase09_executors_and_approval.py`` — plain ``def test_*`` functions
that drive coroutines through :func:`asyncio.run` (no ``pytest.mark.asyncio``).

Pre-persist guarantees (ac-007) are asserted by calling ``stage.run(ctx)``
DIRECTLY rather than through ``run_stage_bracketed`` so the typed
``StageExecutionError`` raised by the gate propagates unwrapped, exactly as it
would in the bracket's plain-exception arm (see
``test_stage_bracket.py::test_bracket_plain_failure_wraps_in_stage_execution_error``).

BoundTask decision (reported to caller): a real
:class:`molexp.harness.schemas.bound_workflow.BoundTask` is cheap to build
(``parameters`` / ``inputs`` / ``outputs`` accept empty dicts), so ac-003 uses
a genuine ``BoundTask`` — the strongest "handled identically" assertion — and
no local stub is needed.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from molexp.harness import CapabilityRegistry, LocalExecutor, StageExecutionError
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.policy import enforce_side_effect_approvals, make_side_effect_approval_requests
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.schemas import CapabilityInvocationResult, PlanArtifactRef, ToolCapability
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.schemas.bound_workflow import BoundTask
from molexp.harness.stages import Approver, InvokeCapability, auto_grant_approver
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

# ─────────────────────────────────────────────────────────── constants / paths

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ECHO_PATH = "tests.test_harness._capability_fixtures:echo"


# ──────────────────────────────────────────────────────────── fixtures / helpers


def _make_capability(
    *,
    id_: str = "cap.invoke",
    name: str = "Invoke",
    description: str = "An invokable capability",
    package: str = "tests",
    callable_path: str | None = None,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    side_effects: list[str] | None = None,
) -> ToolCapability:
    """Build a ``ToolCapability`` (factory copied from ``test_invoke_capability_stage.py``)."""
    if properties is None:
        properties = {"message": {"type": "string"}}
    if required is None:
        required = ["message"]
    return ToolCapability(
        id=id_,
        package=package,
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties, "required": required},
        output_schema={"type": "object", "properties": {}},
        callable_path=callable_path,
        side_effects=side_effects or [],
    )


def _make_bound_task(*, id_: str, side_effects: list[str]) -> BoundTask:
    """Build a minimal-but-real ``BoundTask`` exposing ``id`` + ``side_effects``."""
    return BoundTask(
        id=id_,
        ir_task_id=f"ir-{id_}",
        capability_id="cap.bound",
        package="tests",
        callable="tests.mod:fn",
        parameters={},
        inputs={},
        outputs={},
        side_effects=side_effects,
    )


def _make_ctx(
    root: Path,
    *,
    registry: CapabilityRegistry | None = None,
    run_id: str = "run-side-effect",
) -> HarnessRunContext:
    """Build a fresh ``HarnessRunContext`` backed by isolated on-disk stores.

    Mirrors the ``_make_ctx`` helper in ``test_invoke_capability_stage.py``.
    """
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id=run_id,
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
        capability_registry=registry,
    )


async def denying_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Reject every request — used to prove the gate aborts before dispatch."""
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="test",
        decided_at=datetime.now(tz=UTC),
        reason="denied",
    )


# A type-checked binding: the static checker enforces that ``denying_approver``
# conforms to the ``Approver`` callback contract (this is also what makes the
# ``Approver`` import load-bearing rather than decorative).
_DENYING_APPROVER: Approver = denying_approver


@pytest.fixture()
def subprocess_can_import_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed ``PYTHONPATH`` with the repo root so a ``LocalExecutor`` subprocess
    can ``import tests.test_harness._capability_fixtures`` (copied from
    ``test_invoke_capability_stage.py``)."""
    existing = os.environ.get("PYTHONPATH", "")
    combined = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", combined)


# ──────────────────────────────────── ac-002 · pure derivation, destructive cap
# Category: basics + domain validation (the exact request shape the gate gates on).


def test_destructive_capability_yields_one_overwrite_request() -> None:
    """A destructive capability emits exactly one fully-specified ``overwrite`` request."""
    cap = _make_capability(id_="cap.delete", side_effects=["delete", "write"])

    requests = make_side_effect_approval_requests([cap])

    assert len(requests) == 1
    (request,) = requests
    assert request.intent == "overwrite"
    assert request.triggered_by_policy == "side_effects_present"
    assert cap.id in request.reason
    assert request.metadata == {
        "capability_id": cap.id,
        "side_effects": ["delete", "write"],
    }


# ──────────────────────────────── ac-003 · pure derivation, mixed list + BoundTask
# Category: edge cases + integration (read-only omitted; dedup+sort; BoundTask parity).


def test_mixed_items_emit_only_for_destructive_with_dedup_sorted_metadata() -> None:
    """Across a mixed list, only the destructive items (caps + a real ``BoundTask``)
    emit requests, each with deduped + sorted ``side_effects`` metadata."""
    ro1 = _make_capability(id_="cap.ro1", side_effects=[])
    ro2 = _make_capability(id_="cap.ro2", side_effects=[])
    d1 = _make_capability(id_="cap.d1", side_effects=["delete"])
    # Duplicate + unsorted on purpose: exercises sorted(set(...)).
    d2 = _make_capability(id_="cap.d2", side_effects=["write", "delete", "write"])
    bound = _make_bound_task(id_="task.d3", side_effects=["network", "delete"])
    items: list[ToolCapability | BoundTask] = [ro1, d1, ro2, d2, bound]

    requests = make_side_effect_approval_requests(items)

    assert len(requests) == 3  # exactly the destructive item count
    by_id = {r.metadata["capability_id"]: r for r in requests}
    assert set(by_id) == {"cap.d1", "cap.d2", "task.d3"}  # both read-only omitted
    assert by_id["cap.d2"].metadata["side_effects"] == ["delete", "write"]
    assert by_id["task.d3"].metadata["side_effects"] == ["delete", "network"]
    # The BoundTask is handled identically to a ToolCapability.
    assert by_id["task.d3"].intent == "overwrite"
    assert by_id["task.d3"].triggered_by_policy == "side_effects_present"


# ────────────────────────────────────────── ac-004 · runtime bypass (no side effects)
# Category: lifecycle (true bypass: no gate, no artifact, returns None).


def test_enforce_bypasses_when_no_side_effects(tmp_path: Path) -> None:
    """With no side-effecting items the call returns ``None`` and persists nothing."""
    cap = _make_capability(id_="cap.ro", side_effects=[])
    ctx = _make_ctx(tmp_path)

    result = asyncio.run(enforce_side_effect_approvals([cap], ctx=ctx))

    assert result is None
    assert ctx.artifact_store.list_by_kind("side_effect_approval") == []


# ─────────────────────────────────────────────── ac-005 · runtime denial propagates
# Category: edge case + integration (denial raises; both audit events logged in order).


def test_enforce_raises_and_logs_request_then_rejection_on_denial(tmp_path: Path) -> None:
    """A denied destructive item raises ``StageExecutionError`` and the event log
    records ``approval_requested`` before ``approval_rejected``."""
    cap = _make_capability(id_="cap.del", side_effects=["delete"])
    ctx = _make_ctx(tmp_path)

    with pytest.raises(StageExecutionError):
        asyncio.run(enforce_side_effect_approvals([cap], ctx=ctx, approve=_DENYING_APPROVER))

    types = [e.type for e in ctx.event_log.list_events(ctx.run_id)]
    assert "approval_requested" in types
    assert "approval_rejected" in types
    assert types.index("approval_requested") < types.index("approval_rejected")


# ──────────────────────────────────────────── ac-006 · runtime grant returns summary
# Category: basics (granted gate persists a kind-tagged summary PlanArtifactRef).


def test_enforce_returns_side_effect_approval_artifact_on_grant(tmp_path: Path) -> None:
    """An auto-granted destructive item yields a ``side_effect_approval`` ``PlanArtifactRef``."""
    cap = _make_capability(id_="cap.del", side_effects=["delete"])
    ctx = _make_ctx(tmp_path)

    ref = asyncio.run(enforce_side_effect_approvals([cap], ctx=ctx, approve=auto_grant_approver))

    assert ref is not None
    assert isinstance(ref, PlanArtifactRef)
    assert ref.kind == "side_effect_approval"


# ──────────────────────────────────────── ac-007 · InvokeCapability wiring, denied
# Category: integration (destructive + denier → never dispatched, no result artifact).


def test_destructive_capability_denied_never_dispatches(tmp_path: Path) -> None:
    """A destructive ``InvokeCapability`` with a denying approver aborts before the
    executor runs — no ``capability_invocation_result`` is ever persisted."""
    cap = _make_capability(id_="cap.del", callable_path=_ECHO_PATH, side_effects=["delete"])
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability(
        "cap.del",
        {"message": "hi"},
        executor=LocalExecutor(),
        approve=_DENYING_APPROVER,
    )

    with pytest.raises(StageExecutionError):
        asyncio.run(stage.run(ctx))

    assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []


# ────────────────────────────────────── ac-007 · InvokeCapability wiring, read-only
# Category: integration (read-only bypasses the gate even with a denier; dispatches).


def test_read_only_capability_bypasses_gate_and_dispatches(
    tmp_path: Path,
    subprocess_can_import_fixtures: None,
) -> None:
    """A read-only ``InvokeCapability`` bypasses the gate (a denier never fires) and
    dispatches normally, persisting a succeeded ``capability_invocation_result``."""
    cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH, side_effects=[])
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability(
        "cap.echo",
        {"message": "hi"},
        executor=LocalExecutor(),
        approve=_DENYING_APPROVER,
    )

    ref = asyncio.run(stage.run(ctx))

    assert ref.kind == "capability_invocation_result"
    result = CapabilityInvocationResult.model_validate_json(ctx.artifact_store.get(ref.id))
    assert result.status == "succeeded"
