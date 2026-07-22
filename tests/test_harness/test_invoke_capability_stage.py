"""Tests for the ``InvokeCapability`` direct-invocation stage + its resolver.

Two src units are covered (no dedicated resolver test file exists):

- :func:`molexp.harness.capability.resolve_callable` — resolves a
  ``"module.path:attr"`` (preferred) or ``"module.path.attr"`` (dotted-last)
  reference to a callable, raising :class:`CapabilityResolutionError` on every
  failure mode with **no** silent fallback.
- :class:`molexp.harness.stages.InvokeCapability` — validates the call,
  fail-fast resolves the target callable, persists params, materializes +
  runs an ``invoke_capability.py`` runner through an injected :class:`Executor`,
  and lifts the :class:`CommandResult` into a persisted
  :class:`CapabilityInvocationResult`.

Bracket note: ``run_stage_bracketed`` re-wraps any non-persisted failure in
:class:`StageExecutionError` (see ``test_stage_bracket.py``), so the
"raises before persisting" cases call ``stage.run(ctx)`` directly to assert
the *specific* typed error; the success/lineage case drives the full bracket.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from molexp.harness import (
    CapabilityRegistry,
    DryRunExecutor,
    LocalExecutor,
    StageExecutionError,
)
from molexp.harness.capability import resolve_callable
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage_runner import run_stage_bracketed
from molexp.harness.errors import (
    CapabilityCallValidationError,
    CapabilityResolutionError,
)
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.schemas import CapabilityInvocationResult, ToolCapability
from molexp.harness.stages import InvokeCapability
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from tests.test_harness._capability_fixtures import echo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ECHO_PATH = "tests.test_harness._capability_fixtures:echo"

# ──────────────────────────────────────────────────────────── fixtures / helpers


def _make_capability(
    *,
    id_: str = "cap.invoke",
    callable_path: str | None = None,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    side_effects: list[str] | None = None,
) -> ToolCapability:
    """Build a ``ToolCapability`` with an explicit ``callable_path``."""
    if properties is None:
        properties = {"message": {"type": "string"}}
    if required is None:
        required = ["message"]
    return ToolCapability(
        id=id_,
        package="tests",
        name="Invoke",
        description="An invokable capability",
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
    """Seed ``PYTHONPATH`` with the repo root so the ``LocalExecutor`` child can
    ``import tests.test_harness._capability_fixtures`` (a fresh subprocess
    inherits the parent environment, not pytest's in-process ``sys.path``)."""
    existing = os.environ.get("PYTHONPATH", "")
    combined = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", combined)


class TestResolveCallable:
    """``resolve_callable`` — both reference spellings, no-fallback rejection."""

    @pytest.mark.parametrize(
        "path",
        [_ECHO_PATH, "tests.test_harness._capability_fixtures.echo"],
        ids=["colon-form", "dotted-last-form"],
    )
    def test_resolves_both_reference_spellings(self, path: str) -> None:
        """``module:attr`` and ``module.path.attr`` both resolve the live callable."""
        assert resolve_callable(path) is echo

    @pytest.mark.parametrize(
        "bad_path",
        [
            None,
            "",
            "nonexistent.module:x",
            "tests.test_harness._capability_fixtures:does_not_exist",
            "tests.test_harness._capability_fixtures:NOT_CALLABLE",
        ],
        ids=["none", "empty-string", "unimportable-module", "missing-attribute", "not-callable"],
    )
    def test_rejects_every_invalid_path_without_fallback(self, bad_path: str | None) -> None:
        """Each documented failure mode raises ``CapabilityResolutionError``."""
        with pytest.raises(CapabilityResolutionError):
            resolve_callable(bad_path)


class TestInvokeCapability:
    """The ``InvokeCapability`` stage: guard order, executor dispatch, lineage."""

    def test_missing_registry_raises_stage_execution_error(self, tmp_path: Path) -> None:
        """``ctx.capability_registry is None`` aborts with ``StageExecutionError``."""
        ctx = _make_ctx(tmp_path, registry=None)
        stage = InvokeCapability("cap.x", {"message": "hi"})

        with pytest.raises(StageExecutionError):
            asyncio.run(stage.run(ctx))

    def test_invalid_params_raise_before_persisting(self, tmp_path: Path) -> None:
        """Schema-invalid params raise ``CapabilityCallValidationError``, persist nothing."""
        cap = _make_capability(id_="cap.x", callable_path=_ECHO_PATH)
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
        stage = InvokeCapability("cap.x", {"unexpected": "x"})

        with pytest.raises(CapabilityCallValidationError):
            asyncio.run(stage.run(ctx))

        assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []
        assert ctx.artifact_store.list_by_kind("capability_invocation_params") == []

    def test_unresolvable_callable_raises_before_persisting(self, tmp_path: Path) -> None:
        """The fail-fast resolve guard rejects an unresolvable ``callable_path``
        (raises ``CapabilityResolutionError``) before any artifact is written."""
        # required=[] lets the empty params dict pass validation, so execution
        # reaches the resolve guard.
        cap = _make_capability(id_="cap.u", callable_path="nope.nope:x", required=[])
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
        stage = InvokeCapability("cap.u", {})

        with pytest.raises(CapabilityResolutionError):
            asyncio.run(stage.run(ctx))

        assert ctx.artifact_store.list_by_kind("capability_invocation_result") == []
        assert ctx.artifact_store.list_by_kind("capability_invocation_params") == []

    def test_local_executor_success_persists_result_and_stamps_lineage(
        self,
        tmp_path: Path,
        subprocess_can_import_fixtures: None,
    ) -> None:
        """A successful call persists a ``succeeded`` result carrying the runner's
        ``{"return": <value>}`` payload, linked ``derived_from`` its params
        artifact with the producing stage + run id stamped by the bracket."""
        cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH)
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]), run_id="run-lineage")
        stage = InvokeCapability("cap.echo", {"message": "hi"}, executor=LocalExecutor())

        result_ref = asyncio.run(run_stage_bracketed(ctx, stage))

        assert result_ref.kind == "capability_invocation_result"
        result = CapabilityInvocationResult.model_validate_json(
            ctx.artifact_store.get(result_ref.id)
        )
        assert result.status == "succeeded"
        assert result.exit_code == 0
        assert result.outputs == {"return": {"echoed": {"message": "hi"}}}

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

    def test_dry_run_executor_is_a_no_op(self, tmp_path: Path) -> None:
        """``DryRunExecutor`` succeeds with empty ``outputs`` + its executor stamp;
        the real callable never runs."""
        cap = _make_capability(id_="cap.echo", callable_path=_ECHO_PATH)
        ctx = _make_ctx(tmp_path, registry=InMemoryCapabilityRegistry([cap]))
        stage = InvokeCapability("cap.echo", {"message": "hi"}, executor=DryRunExecutor())

        ref = asyncio.run(run_stage_bracketed(ctx, stage))

        result = CapabilityInvocationResult.model_validate_json(ctx.artifact_store.get(ref.id))
        assert result.status == "succeeded"
        assert result.exit_code == 0
        assert result.outputs == {}
        assert result.metadata["executor"] == "DryRunExecutor"
