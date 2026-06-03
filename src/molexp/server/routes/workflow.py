"""Workflow document write-back route.

A thin HTTP wrapper over :class:`~molexp.workflow.codec.WorkflowCodec`: the
free-layout canvas PUTs an edited workflow IR document, the route validates
it through ``ir_to_spec`` (so an invalid document surfaces as a structured
4xx via the workflow-layer ``WorkflowError`` / ``ValueError`` handlers rather
than a 500), normalizes it through ``spec_to_ir``, and persists it onto the
experiment's ``workflow_source`` metadata via the atomic ``experiment.save()``.

The route never re-implements IR parsing — the codec is the single owner.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from molexp.workflow import default_codec
from molexp.workspace import (
    ExperimentNotFoundError as WorkspaceExperimentNotFoundError,
)
from molexp.workspace import (
    ProjectNotFoundError as WorkspaceProjectNotFoundError,
)

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError
from ..schemas import WorkflowDocumentRequest, WorkflowDocumentResponse

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/workflow",
    tags=["workflow"],
)


def _resolve_experiment(workspace, project_id: str, experiment_id: str):  # noqa: ANN001, ANN202
    """Strict-getter chain — returns the workspace ``Experiment`` or raises 404."""
    try:
        project = workspace.get_project(project_id)
    except WorkspaceProjectNotFoundError as exc:
        raise ExperimentNotFoundError(experiment_id) from exc
    try:
        return project.get_experiment(experiment_id)
    except WorkspaceExperimentNotFoundError as exc:
        raise ExperimentNotFoundError(experiment_id) from exc


def _normalize(document: dict) -> dict:
    """Validate + round-trip an IR document through the codec.

    ``ir_to_spec`` raises ``WorkflowError`` (cycles, unknown tasks, …) or
    ``ValueError`` (malformed IR) on invalid input — both are mapped to a
    structured 4xx by the registered exception handlers, never a 500.
    """
    spec = default_codec.ir_to_spec(document)
    return dict(default_codec.spec_to_ir(spec))


@router.put("", response_model=WorkflowDocumentResponse)
def put_workflow_document(
    project_id: str,
    experiment_id: str,
    payload: WorkflowDocumentRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> WorkflowDocumentResponse:
    """Validate, normalize, and persist an edited workflow IR document."""
    experiment = _resolve_experiment(workspace, project_id, experiment_id)
    normalized = _normalize(payload.document)

    experiment.metadata = experiment.metadata.model_copy(
        update={"workflow_source": json.dumps(normalized, sort_keys=True)}
    )
    experiment.save()

    return WorkflowDocumentResponse(
        project_id=project_id,
        experiment_id=experiment_id,
        document=normalized,
    )


@router.get("", response_model=WorkflowDocumentResponse)
def get_workflow_document(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> WorkflowDocumentResponse:
    """Return the persisted workflow IR document, or 404 if none stored."""
    experiment = _resolve_experiment(workspace, project_id, experiment_id)
    source = experiment.metadata.workflow_source
    if not source:
        raise ExperimentNotFoundError(f"{experiment_id} (no workflow document)")
    try:
        document = json.loads(source)
    except (ValueError, TypeError) as exc:
        # Stored source is not an IR JSON document (e.g. legacy Python source).
        raise ExperimentNotFoundError(f"{experiment_id} (no workflow document)") from exc
    if not isinstance(document, dict):
        raise ExperimentNotFoundError(f"{experiment_id} (no workflow document)")

    return WorkflowDocumentResponse(
        project_id=project_id,
        experiment_id=experiment_id,
        document=document,
    )
