"""Compute-target registry routes — CRUD on workspace ComputeTargets.

These endpoints are the UI/API counterpart of ``molexp target add/remove/list/test``.
A target is the (Transport × Scheduler) destination a run can be submitted to.
The workspace (``workspace.json``) is the source of truth; this router is a
thin adapter on top of :mod:`molexp.workspace.targets`.
"""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException

from molexp.workspace import (
    ComputeTarget,
    Workspace,
    add_target,
    get_target,
    list_targets,
    remove_target,
    to_transport,
)

from ..dependencies import get_workspace
from ..schemas import (
    TargetCreateRequest,
    TargetListResponse,
    TargetResponse,
    TargetTestCheck,
    TargetTestResponse,
)

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get("", response_model=TargetListResponse)
def list_targets_endpoint(
    workspace: Workspace = Depends(get_workspace),
) -> TargetListResponse:
    """List every compute target registered on the workspace."""
    targets = list_targets(workspace)
    rows = [TargetResponse.from_model(t) for t in targets]
    return TargetListResponse(targets=rows, total=len(rows))


@router.post("", response_model=TargetResponse, status_code=201)
def create_target_endpoint(
    payload: TargetCreateRequest,
    workspace: Workspace = Depends(get_workspace),
) -> TargetResponse:
    """Register a new compute target.

    Mirrors ``molexp target add NAME --scratch ... [--host ...] [--scheduler ...]``.
    """
    try:
        target = ComputeTarget(
            name=payload.name,
            host=payload.host,
            port=payload.port,
            identity_file=payload.identity_file,
            ssh_opts=list(payload.ssh_opts),
            scheduler=payload.scheduler,
            scratch_root=payload.scratch_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        add_target(workspace, target)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return TargetResponse.from_model(target)


@router.delete("/{name}", status_code=204)
def delete_target_endpoint(
    name: str,
    workspace: Workspace = Depends(get_workspace),
) -> None:
    """Remove the named compute target from the workspace registry."""
    try:
        remove_target(workspace, name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{name}/test", response_model=TargetTestResponse)
def test_target_endpoint(
    name: str,
    workspace: Workspace = Depends(get_workspace),
) -> TargetTestResponse:
    """Verify connectivity to a target — runs the same round-trip probe as
    ``molexp target test`` (true / mkdir scratch / 1-byte file round-trip).

    Returns ``ok=False`` with the failing step's detail rather than raising,
    so the UI can render the failure inline.
    """
    try:
        target = get_target(workspace, name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    checks: list[TargetTestCheck] = []

    if target.is_remote:
        if shutil.which("ssh") is None:
            return TargetTestResponse(
                name=name,
                ok=False,
                checks=[],
                error="ssh binary not found in PATH on the server",
            )
        if shutil.which("rsync") is None:
            return TargetTestResponse(
                name=name,
                ok=False,
                checks=[],
                error="rsync binary not found in PATH on the server",
            )

    transport = to_transport(target)

    # 1. trivial command.
    try:
        result = transport.run(["true"], timeout=15)
        if result.returncode != 0:
            checks.append(
                TargetTestCheck(
                    label="command execution",
                    ok=False,
                    detail=f"exit={result.returncode}: {result.stderr}",
                )
            )
            return TargetTestResponse(
                name=name, ok=False, checks=checks, error="command execution failed"
            )
        checks.append(TargetTestCheck(label="command execution", ok=True))
    except Exception as exc:
        return TargetTestResponse(
            name=name, ok=False, checks=checks, error=f"transport.run failed: {exc}"
        )

    # 2. mkdir scratch.
    try:
        transport.mkdir(target.scratch_root, parents=True, exist_ok=True)
        checks.append(
            TargetTestCheck(label=f"mkdir {target.scratch_root}", ok=True)
        )
    except Exception as exc:
        checks.append(
            TargetTestCheck(
                label=f"mkdir {target.scratch_root}",
                ok=False,
                detail=str(exc),
            )
        )
        return TargetTestResponse(
            name=name, ok=False, checks=checks, error=f"mkdir failed: {exc}"
        )

    # 3. file round-trip.
    probe = f"{target.scratch_root.rstrip('/')}/.molexp-target-test"
    try:
        transport.write_text(probe, "x")
        if transport.read_text(probe) != "x":
            checks.append(
                TargetTestCheck(
                    label="file round-trip", ok=False, detail="content mismatch"
                )
            )
            return TargetTestResponse(
                name=name, ok=False, checks=checks, error="file round-trip mismatch"
            )
        transport.remove(probe)
        checks.append(TargetTestCheck(label="file round-trip", ok=True))
    except Exception as exc:
        checks.append(
            TargetTestCheck(label="file round-trip", ok=False, detail=str(exc))
        )
        return TargetTestResponse(
            name=name, ok=False, checks=checks, error=f"round-trip failed: {exc}"
        )

    return TargetTestResponse(name=name, ok=True, checks=checks, error=None)
