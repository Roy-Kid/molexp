"""Curate-task routes — the shared curation flow as a background server task.

Drives ``/api/projects/{p}/experiments/{e}/curate-tasks`` with a stub
``curation_planner`` gateway (canned :class:`CurationInvocation`) and a stubbed
merged-registry seam, so the full flow runs offline in the app's event loop.
Asserts:

- **ac-002/ac-003 (happy)** POST starts a task, the background run reaches
  ``completed`` and reports the selected capability.
- **ac-006** the curate Run is content-addressed on the curate-mode key — the
  same request reuses the same Run id (== ``derive_run_id``).
- **ac-003 (parity)** a destructive ``move_run`` driven over HTTP leaves the
  workspace in the *same* on-disk state (same relocated entity + same artifact
  kinds) as the identical request driven directly from Python — the Python≡UI
  invariant for curation.
- **ac-005 (route)** the route is a thin adapter: it delegates to the single
  ``run_curation_flow`` exactly once with request-derived arguments.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from molexp.harness import InMemoryCapabilityRegistry, auto_grant_approver
from molexp.harness.capabilities import curation_capabilities
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.server.app import create_app
from molexp.server.curate_runtime import gateway as curate_gateway
from molexp.server.dependencies import get_workspace
from molexp.workspace.utils import derive_run_id

_BASE = "/api/projects/test-project/experiments/curate-exp/curate-tasks"
_ARTIFACT_KINDS = ("prompt", "capability_catalog", "capability_invocation_result")


def _scan_factory(run: Any, model: str) -> Any:
    """Gateway whose ``curation_planner`` picks the read-only ``scan_workspace``."""
    gw = StubAgentGateway(FileArtifactStore(root=Path(run.run_dir) / "artifacts"))
    gw.register(
        "curation_planner",
        {"capability_id": "molexp.curation.scan_workspace", "references": {}, "reason": "scan"},
        output_kind="curation_invocation",
    )
    return gw


def _move_factory(run: Any, model: str) -> Any:
    """Gateway whose ``curation_planner`` relocates ``subject`` into ``target-exp``."""
    gw = StubAgentGateway(FileArtifactStore(root=Path(run.run_dir) / "artifacts"))
    gw.register(
        "curation_planner",
        {
            "capability_id": "molexp.curation.move_run",
            "references": {"run": "subject", "target_experiment": "target-exp"},
            "reason": "relocate",
        },
        output_kind="curation_invocation",
    )
    return gw


@pytest.fixture(autouse=True)
def _patched_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the merged-registry seam so molmcp is never spawned in route tests."""

    async def _stub_registry(workspace_root: str, **_: Any) -> InMemoryCapabilityRegistry:
        return InMemoryCapabilityRegistry(curation_capabilities())

    monkeypatch.setattr(
        "molexp.server.curate_runtime.flow.aresolve_curation_capability_registry",
        _stub_registry,
    )


@pytest.fixture
def curate_client(workspace: Any, project: Any) -> Iterator[TestClient]:
    project.add_experiment("curate-exp")
    project.add_experiment("target-exp")
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    with TestClient(app) as client:  # context manager → lifespan cancels tasks on teardown
        yield client
    curate_gateway.reset_curate_gateway_factory()


def _await_terminal(client: TestClient, task_id: str, tries: int = 200) -> dict[str, Any]:
    """Poll the task until it leaves ``running`` (gives the bg loop time to run)."""
    url = f"{_BASE}/{task_id}"
    body: dict[str, Any] = client.get(url).json()
    for _ in range(tries):
        if body["status"] != "running":
            return body
        time.sleep(0.05)
        body = client.get(url).json()
    return body


def test_create_curate_task_runs_flow(curate_client: TestClient) -> None:
    """ac-002/ac-003 — POST starts a task; the flow completes and reports the cap."""
    curate_gateway.set_curate_gateway_factory(_scan_factory)

    resp = curate_client.post(_BASE, json={"request": "inventory the workspace", "model": "m"})
    assert resp.status_code == 201, resp.text
    started = resp.json()
    assert started["status"] == "running"
    assert started["runId"]

    final = _await_terminal(curate_client, started["taskId"])
    assert final["status"] == "completed", final
    assert final["capabilityId"] == "molexp.curation.scan_workspace"
    assert final["granted"] is True

    listed = curate_client.get(_BASE).json()
    assert any(t["taskId"] == started["taskId"] for t in listed["tasks"])


def test_curate_run_is_content_addressed(curate_client: TestClient) -> None:
    """ac-006 — same request → same run id, equal to derive_run_id of the key."""
    curate_gateway.set_curate_gateway_factory(_scan_factory)
    request_text = "aggregate run outputs by kind"
    payload = {"request": request_text, "model": "m"}

    first = curate_client.post(_BASE, json=payload).json()
    _await_terminal(curate_client, first["taskId"])
    second = curate_client.post(_BASE, json=payload).json()

    expected = derive_run_id({"mode": "curate", "request": request_text})
    assert first["runId"] == expected
    assert second["runId"] == first["runId"]


def test_get_unknown_curate_task_returns_404(curate_client: TestClient) -> None:
    assert curate_client.get(f"{_BASE}/curate-does-not-exist").status_code == 404


def test_route_matches_python_path_for_destructive_move(
    curate_client: TestClient,
    workspace: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """ac-003 (parity) — a ``move_run`` driven over HTTP and the identical request
    driven directly from Python converge on the same on-disk state and the same
    artifact kinds."""
    from molexp.server.curate_runtime.flow import run_curation_flow
    from molexp.workspace import Workspace

    curate_gateway.set_curate_gateway_factory(_move_factory)
    request_text = "move subject run to target-exp"

    # ── Route path ──────────────────────────────────────────────────────────
    proj = workspace.get_project("test-project")
    proj.get_experiment("curate-exp").add_run({"seed": 0}, id="subject")
    started = curate_client.post(_BASE, json={"request": request_text, "model": "m"}).json()
    final = _await_terminal(curate_client, started["taskId"])
    assert final["status"] == "completed", final

    route_source = {r.id for r in proj.get_experiment("curate-exp").list_runs()}
    route_target = {r.id for r in proj.get_experiment("target-exp").list_runs()}
    assert "subject" not in route_source, "HTTP move must remove the run from its source"
    assert "subject" in route_target, "HTTP move must relocate the run to the target"

    # ── Python path on a twin workspace ──────────────────────────────────────
    ws2 = Workspace(root=tmp_path_factory.mktemp("py-twin"), name="Test")
    proj2 = ws2.add_project("test-project")
    curate_exp2 = proj2.add_experiment("curate-exp")
    proj2.add_experiment("target-exp")
    curate_exp2.add_run({"seed": 0}, id="subject")
    params = {"mode": "curate", "request": request_text}
    run2 = curate_exp2.add_run(params, id=derive_run_id(params))
    asyncio.run(
        run_curation_flow(
            request_text,
            workspace=ws2,
            experiment=curate_exp2,
            run=run2,
            gateway=_move_factory(run2, "m"),
            approve=auto_grant_approver,
        )
    )
    py_source = {r.id for r in proj2.get_experiment("curate-exp").list_runs()}
    py_target = {r.id for r in proj2.get_experiment("target-exp").list_runs()}

    # Parity: identical mutation outcome for the subject run.
    assert ("subject" in route_source) == ("subject" in py_source)
    assert ("subject" in route_target) == ("subject" in py_target)

    # Parity: the same artifact kinds are persisted under both runs.
    route_run = proj.get_experiment("curate-exp").get_run(derive_run_id(params))
    route_store = FileArtifactStore(root=Path(route_run.run_dir) / "artifacts")
    py_store = FileArtifactStore(root=Path(run2.run_dir) / "artifacts")
    for kind in _ARTIFACT_KINDS:
        assert route_store.list_by_kind(kind), f"route missing {kind!r}"
        assert py_store.list_by_kind(kind), f"python missing {kind!r}"


def test_route_is_thin_adapter_over_shared_flow(
    curate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ac-005 (route) — the route delegates to run_curation_flow exactly once with
    request-derived arguments (no discover/select/invoke logic of its own)."""
    from molexp.server.curate_runtime.flow import CurationResult

    curate_gateway.set_curate_gateway_factory(_scan_factory)
    recorded: list[str] = []

    async def _rec(
        request: str,
        *,
        workspace: Any,
        experiment: Any,
        run: Any,
        gateway: Any,
        approve: Any = None,
    ) -> CurationResult:
        recorded.append(request)
        return CurationResult(
            capability_id="molexp.curation.scan_workspace",
            mutation_summary="queried (read-only)",
            granted=True,
            artifact_ids=[],
        )

    monkeypatch.setattr("molexp.server.curate_runtime.flow.run_curation_flow", _rec)

    started = curate_client.post(_BASE, json={"request": "list everything", "model": "m"}).json()
    final = _await_terminal(curate_client, started["taskId"])

    assert final["status"] == "completed", final
    assert recorded == ["list everything"], "shared flow must be called exactly once"
    assert final["capabilityId"] == "molexp.curation.scan_workspace"
