"""Tests for the ``InvokeCapability`` direct-invocation stage (workspace-curation-toolset, link 02).

Contract under test (RED — the symbols do not exist yet, so this module
fails at *import/collection* until the production code lands):

- :func:`molexp.harness.resolve_callable` resolves a ``"module.path:attr"``
  (preferred) or ``"module.path.attr"`` (dotted-last) reference to a callable,
  and raises :class:`molexp.harness.CapabilityResolutionError` on every
  failure mode with no silent fallback.
- :class:`molexp.harness.CapabilityInvocationResult` is the frozen pydantic
  record mirroring ``ExecutionResult`` for a single capability call.
- :class:`molexp.harness.InvokeCapability` is a :class:`Stage` that validates
  the call, fail-fast resolves the target callable, persists the params,
  materializes + runs an ``invoke_capability.py`` runner through an injected
  :class:`Executor`, and lifts the :class:`CommandResult` into a persisted
  :class:`CapabilityInvocationResult`.

Resolved contract ambiguity (reported to the caller): the audit bracket
``run_stage_bracketed`` re-wraps any non-``StagePersistedFailureError`` in
:class:`StageExecutionError` (see
``test_stage_bracket.py::test_bracket_plain_failure_wraps_in_stage_execution_error``).
A bracket-level ``pytest.raises(CapabilityCallValidationError)`` /
``pytest.raises(CapabilityResolutionError)`` would therefore never match the
*specific* typed error. To assert the precise type the spec calls for, the
"raises before persisting" cases (ac-002, ac-005) invoke ``stage.run(ctx)``
directly — which is also the exact surface those guarantees describe — while
the success/lineage cases (ac-003, ac-004, ac-006) drive the full bracket.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from molexp.harness import (
    CapabilityCallValidationError,
    CapabilityInvocationResult,
    CapabilityRegistry,
    CapabilityResolutionError,
    DryRunExecutor,
    HarnessError,
    InMemoryCapabilityRegistry,
    InvokeCapability,
    LocalExecutor,
    Stage,
    StageExecutionError,
    ToolCapability,
    resolve_callable,
)
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage_runner import run_stage_bracketed
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from tests.test_harness._capability_fixtures import echo

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
    """Build a ``ToolCapability`` with an explicit ``callable_path``.

    Mirrors the factory in ``test_in_memory_capability_registry.py`` but adds
    the ``callable_path`` parameter the direct-invocation path depends on.
    """
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


def _make_ctx(
    root: Path,
    *,
    registry: CapabilityRegistry | None = None,
    run_id: str = "run-cap",
) -> HarnessRunContext:
    """Build a fresh ``HarnessRunContext`` backed by isolated on-disk stores."""
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


@pytest.fixture()
def subprocess_can_import_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the repo root on ``PYTHONPATH`` so the ``LocalExecutor`` subprocess
    can ``import tests.test_harness._capability_fixtures``.

    pytest makes the repo root importable in-process (sys.path manipulation),
    but a fresh ``[sys.executable, "invoke_capability.py"]`` child inherits the
    parent *environment*, not its ``sys.path``. Seeding ``PYTHONPATH`` keeps
    the subprocess hermetic and deterministic regardless of how pytest itself
    was launched.
    """
    existing = os.environ.get("PYTHONPATH", "")
    combined = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", combined)


# ───────────────────────────────────────────────────── ac-001 · resolve_callable
# Category: basics + edge cases (every documented failure mode, no fallback).


def test_resolve_callable_colon_form_returns_the_callable() -> None:
    """The preferred ``module:attr`` form resolves to the live callable."""
    resolved = resolve_callable(_ECHO_PATH)
    assert resolved is echo


def test_resolve_callable_dotted_last_form_resolves() -> None:
    """The dotted-last-segment form ``module.path.attr`` also resolves."""
    resolved = resolve_callable("tests.test_harness._capability_fixtures.echo")
    assert resolved is echo


def test_resolve_callable_rejects_none() -> None:
    """``None`` is rejected by the runtime guard (no silent fallback)."""
    with pytest.raises(CapabilityResolutionError):
        resolve_callable(None)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "nonexistent.module:x",
        "tests.test_harness._capability_fixtures:does_not_exist",
        "tests.test_harness._capability_fixtures:NOT_CALLABLE",
    ],
    ids=["empty-string", "unimportable-module", "missing-attribute", "not-callable"],
)
def test_resolve_callable_rejects_invalid_path(bad_path: str) -> None:
    """Each malformed/unresolvable reference raises ``CapabilityResolutionError``."""
    with pytest.raises(CapabilityResolutionError):
        resolve_callable(bad_path)


# ──────────────────────────────────────────────── ac-002 · bad params (pre-persist)
# Category: domain validation / edge case (reject before any artifact is written).


def test_bad_params_raise_before_persisting(tmp_path: Path) -> None:
    """Schema-invalid params raise ``CapabilityCallValidationError`` and persist nothing."""
    cap = _make_capability(id_="cap.x", callable_path=_ECHO_PATH)
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability("cap.x", {"unexpected": "x"})

    with pytest.raises(CapabilityCallValidationError):
        asyncio.run(stage.run(ctx))

    assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []
    assert ctx.artifact_store.list_by_kind("capability_invocation_params") == []


# ───────────────────────────────────────────── ac-003 · LocalExecutor happy path
# Category: basics + integration (real subprocess through the bracket).


def test_local_executor_success_persists_succeeded_result(
    tmp_path: Path,
    subprocess_can_import_fixtures: None,
) -> None:
    """A successful call persists a ``capability_invocation_result`` with the
    runner's ``{"return": <value>}`` payload as ``outputs``."""
    cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH)
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability("cap.echo", {"message": "hi"}, executor=LocalExecutor())

    ref = asyncio.run(run_stage_bracketed(ctx, stage))

    assert ref.kind == "capability_invocation_result"
    result = CapabilityInvocationResult.model_validate_json(ctx.artifact_store.get(ref.id))
    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.outputs == {"return": {"echoed": {"message": "hi"}}}


# ─────────────────────────────────────────────────────────── ac-004 · lineage
# Category: integration (audit lineage edge stamped by the bracket).


def test_lineage_edge_links_result_to_params(
    tmp_path: Path,
    subprocess_can_import_fixtures: None,
) -> None:
    """A ``derived_from`` edge links the result to its params artifact, stamped
    with the producing stage + run id."""
    cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH)
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]), run_id="run-lineage")
    stage = InvokeCapability("cap.echo", {"message": "hi"}, executor=LocalExecutor())

    result_ref = asyncio.run(run_stage_bracketed(ctx, stage))

    params_ref = ctx.artifact_store.latest_by_kind("capability_invocation_params")
    assert params_ref is not None
    edges = ctx.lineage_store.lineage_graph(params_ref.id)["edges"]
    assert {
        "parent_id": params_ref.id,
        "child_id": result_ref.id,
        "relation": "derived_from",
        "stage": "invoke_capability",
        "run_id": "run-lineage",
    } in edges


# ─────────────────────────────────────────── ac-005 · unresolvable (pre-persist)
# Category: edge cases (fail-fast resolve guard before any artifact is written).


@pytest.mark.parametrize(
    "callable_path",
    [None, "nope.nope:x", "tests.test_harness._capability_fixtures:does_not_exist"],
    ids=["callable-path-none", "unimportable-module", "missing-attribute"],
)
def test_unresolvable_callable_raises_before_persisting(
    tmp_path: Path,
    callable_path: str | None,
) -> None:
    """An unresolvable ``callable_path`` raises ``CapabilityResolutionError`` and
    persists no artifacts."""
    # ``required=[]`` + a single optional property means the empty params dict
    # passes ``validate_call``, so execution reaches the resolve guard.
    cap = _make_capability(
        id_="cap.u",
        callable_path=callable_path,
        properties={"message": {"type": "string"}},
        required=[],
    )
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability("cap.u", {})

    with pytest.raises(CapabilityResolutionError):
        asyncio.run(stage.run(ctx))

    assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []
    assert ctx.artifact_store.list_by_kind("capability_invocation_params") == []


# ───────────────────────────────────────────────────────── ac-006 · DryRun no-op
# Category: lifecycle / integration (executor is a no-op; callable never runs).


def test_dry_run_executor_is_a_no_op(tmp_path: Path) -> None:
    """``DryRunExecutor`` succeeds with empty ``outputs`` and its executor stamp;
    the real callable is never invoked (no ``result.json``)."""
    cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH)
    ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
    stage = InvokeCapability("cap.echo", {"message": "hi"}, executor=DryRunExecutor())

    ref = asyncio.run(run_stage_bracketed(ctx, stage))

    result = CapabilityInvocationResult.model_validate_json(ctx.artifact_store.get(ref.id))
    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.outputs == {}
    assert result.metadata["executor"] == "DryRunExecutor"


# ─────────────────────────────────────── edge · missing registry (run() contract)
# Category: edge case (documented step 1: no registry → StageExecutionError).


def test_missing_registry_raises_stage_execution_error(tmp_path: Path) -> None:
    """``ctx.capability_registry is None`` aborts with ``StageExecutionError``."""
    ctx = _make_ctx(tmp_path, registry=None)
    stage = InvokeCapability("cap.x", {"message": "hi"})

    with pytest.raises(StageExecutionError):
        asyncio.run(stage.run(ctx))


# ────────────────────────────────────────────────────────── ac-007 · public surface
# Category: basics (import surface + frozen contracts).


def test_public_surface() -> None:
    """The three new symbols + resolver are reachable from ``molexp.harness`` and
    the artifact kinds are registered."""
    import molexp.harness as h

    assert issubclass(InvokeCapability, Stage)
    assert InvokeCapability.name == "invoke_capability"
    assert issubclass(CapabilityResolutionError, HarnessError)
    assert "capability_invocation_params" in h.WELL_KNOWN_ARTIFACT_KINDS
    assert "capability_invocation_result" in h.WELL_KNOWN_ARTIFACT_KINDS


def test_resolve_callable_re_exported_from_submodule() -> None:
    """``resolve_callable`` is the same object whether imported from the package
    root or its owning ``molexp.harness.capability.resolve`` module."""
    from molexp.harness.capability.resolve import resolve_callable as via_mod

    assert resolve_callable is via_mod


def test_capability_resolution_error_re_exported_from_errors() -> None:
    """``CapabilityResolutionError`` re-export identity holds across modules."""
    from molexp.harness.errors import CapabilityResolutionError as via_mod

    assert CapabilityResolutionError is via_mod
