"""Structured exception hierarchy for MolExp API.

This module defines a consistent exception structure that integrates with
FastAPI's exception handling to provide standardized error responses.
"""

from __future__ import annotations

from typing import Any


class MolExpError(Exception):
    """Base exception for all MolExp API errors.

    Provides structured error information that can be converted to
    consistent HTTP error responses.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (e.g., "NOT_FOUND", "VALIDATION_ERROR")
        status_code: HTTP status code
        details: Optional additional error details
    """

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        error_body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error_body["details"] = self.details
        return {"error": error_body}


# ============================================================================
# Not Found Errors (404)
# ============================================================================


class NotFoundError(MolExpError):
    """Resource not found error."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} '{identifier}' not found",
            code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier},
        )


class ProjectNotFoundError(NotFoundError):
    """Project not found."""

    def __init__(self, id: str) -> None:
        super().__init__("Project", id)


class ExperimentNotFoundError(NotFoundError):
    """Experiment not found."""

    def __init__(self, project_id: str, experiment_id: str | None = None) -> None:
        super().__init__("Experiment", experiment_id or project_id)
        self.details["project_id"] = project_id
        if experiment_id:
            self.details["experiment_id"] = experiment_id


class RunNotFoundError(NotFoundError):
    """Run not found."""

    def __init__(
        self,
        project_id: str,
        experiment_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__("Run", run_id or experiment_id or project_id)
        self.details["project_id"] = project_id
        if experiment_id:
            self.details["experiment_id"] = experiment_id
        if run_id:
            self.details["run_id"] = run_id


class AssetNotFoundError(NotFoundError):
    """Asset not found."""

    def __init__(self, asset_id: str) -> None:
        super().__init__("Asset", asset_id)


class TaskNotFoundError(NotFoundError):
    """Task type not found."""

    def __init__(self, node_id: str) -> None:
        super().__init__("Task", node_id)


class FolderNotFoundError(NotFoundError):
    """Workspace folder not found."""

    def __init__(self, folder_id: str) -> None:
        super().__init__("Folder", folder_id)


class FileNotFoundError(NotFoundError):
    """File not found."""

    def __init__(self, path: str) -> None:
        super().__init__("File", path)


# ============================================================================
# Validation Errors (400)
# ============================================================================


class ValidationError(MolExpError):
    """Request validation error."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=error_details,
        )


class InvalidPathError(ValidationError):
    """Invalid path provided."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            message=f"Invalid path: {reason}",
            details={"path": path, "reason": reason},
        )


class PathNotDirectoryError(ValidationError):
    """Path is not a directory."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message=f"Path is not a directory: {path}",
            details={"path": path},
        )


class PathNotFileError(ValidationError):
    """Path is not a file."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message=f"Path is not a file: {path}",
            details={"path": path},
        )


class FileTooLargeError(ValidationError):
    """File exceeds size limit."""

    def __init__(self, path: str, size: int, max_size: int) -> None:
        super().__init__(
            message=f"File too large: {size} bytes (max: {max_size} bytes)",
            details={"path": path, "size": size, "max_size": max_size},
        )


class BinaryFileError(ValidationError):
    """Binary file not supported for this operation."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message="Binary files not supported for preview",
            details={"path": path},
        )


class InvalidStatusError(ValidationError):
    """Invalid status transition."""

    def __init__(self, current_status: str, requested_status: str) -> None:
        super().__init__(
            message=f"Cannot transition from '{current_status}' to '{requested_status}'",
            details={
                "current_status": current_status,
                "requested_status": requested_status,
            },
        )


# ============================================================================
# Conflict Errors (409)
# ============================================================================


class ConflictError(MolExpError):
    """Resource conflict error."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=409,
            details=details or {},
        )


class DuplicateResourceError(ConflictError):
    """Resource already exists."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} '{identifier}' already exists",
            details={"resource": resource, "identifier": identifier},
        )


class FolderAlreadyAddedError(ConflictError):
    """Folder already added to workspace."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message="Folder already added to workspace",
            details={"path": path},
        )


class PathExistsError(ConflictError):
    """Path already exists."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message=f"Path already exists: {path}",
            details={"path": path},
        )


# ============================================================================
# Permission Errors (403)
# ============================================================================


class PermissionError(MolExpError):
    """Permission denied error."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="PERMISSION_DENIED",
            status_code=403,
            details=details or {},
        )


class PathOutsideWorkspaceError(PermissionError):
    """Path is outside the allowed workspace."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message="Access denied: path outside workspace folder",
            details={"path": path},
        )


# ============================================================================
# Workflow Errors (400/500)
# ============================================================================


class WorkflowError(MolExpError):
    """Base class for workflow-related errors."""


class WorkflowValidationError(WorkflowError):
    """Workflow validation failed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="WORKFLOW_VALIDATION_ERROR",
            status_code=400,
            details=details or {},
        )


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed."""

    def __init__(
        self,
        message: str,
        id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        details = {}
        if id:
            details["id"] = id
        if node_id:
            details["node_id"] = node_id
        super().__init__(
            message=message,
            code="WORKFLOW_EXECUTION_ERROR",
            status_code=500,
            details=details,
        )


class CyclicDependencyError(WorkflowValidationError):
    """Workflow contains a cycle."""

    def __init__(self, tasks: list[str] | None = None) -> None:
        details = {}
        if tasks:
            details["tasks"] = tasks
        super().__init__(
            message="Workflow contains a cycle (circular dependency)",
            details=details,
        )


class TargetTaskNotFoundError(WorkflowValidationError):
    """Target node not found in workflow."""

    def __init__(self, target: str) -> None:
        super().__init__(
            message=f"Target node '{target}' not found in workflow",
            details={"target": target},
        )


# ============================================================================
# Preview Errors (404 / 422)
# ============================================================================
#
# Sidecar-backed dataset preview. A missing / empty / ambiguous / broken
# sidecar is always a typed 4xx — never a 500 — so the UI can distinguish
# "no preview available" from "the preview machinery crashed". These are
# auto-mapped to JSON by the base ``MolExpError`` handler via ``status_code``.


class PreviewSidecarNotFoundError(MolExpError):
    """No same-stem ``.py`` sidecar sits next to the dataset."""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(
            message=f"No preview sidecar found for dataset: {dataset_path}",
            code="PREVIEW_SIDECAR_NOT_FOUND",
            status_code=404,
            details={"dataset_path": dataset_path},
        )


class NoReaderInSidecarError(MolExpError):
    """The sidecar defines no concrete ``BaseTrajectoryReader`` subclass."""

    def __init__(self, sidecar_path: str) -> None:
        super().__init__(
            message=(f"Sidecar defines no molpy.io.BaseTrajectoryReader subclass: {sidecar_path}"),
            code="NO_READER_IN_SIDECAR",
            status_code=422,
            details={"sidecar_path": sidecar_path},
        )


class AmbiguousReaderError(MolExpError):
    """The sidecar defines more than one ``BaseTrajectoryReader`` subclass."""

    def __init__(self, sidecar_path: str, reader_names: list[str]) -> None:
        super().__init__(
            message=(
                f"Sidecar defines {len(reader_names)} BaseTrajectoryReader subclasses "
                f"(expected exactly one): {', '.join(reader_names)}"
            ),
            code="AMBIGUOUS_READER",
            status_code=422,
            details={"sidecar_path": sidecar_path, "readers": reader_names},
        )


class PreviewReaderError(MolExpError):
    """Importing, instantiating, or iterating the sidecar reader failed."""

    def __init__(self, dataset_path: str, reason: str | None = None) -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(
            message=f"Failed to read dataset preview for {dataset_path}{detail}",
            code="PREVIEW_READER_ERROR",
            status_code=422,
            details={"dataset_path": dataset_path, **({"reason": reason} if reason else {})},
        )


# ============================================================================
# Storage Errors (500)
# ============================================================================


class StorageError(MolExpError):
    """Storage operation error."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        path: str | None = None,
    ) -> None:
        details = {}
        if operation:
            details["operation"] = operation
        if path:
            details["path"] = path
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            status_code=500,
            details=details,
        )


# ============================================================================
# Workspace-routing Errors (404 / 405 / 502)
# ============================================================================


class UnknownWorkspaceError(NotFoundError):
    """The ``{ws}`` path segment does not name a served workspace."""

    def __init__(self, key: str) -> None:
        super().__init__("Workspace", key)
        self.code = "UNKNOWN_WORKSPACE"


class RemoteWorkspaceReadOnlyError(MolExpError):
    """A mutating request targeted a remote workspace (read-only in v1)."""

    def __init__(self, key: str) -> None:
        super().__init__(
            message=(
                f"Workspace '{key}' is remote and read-only; "
                "writes (run launch / workflow write-back) are not supported."
            ),
            code="REMOTE_WORKSPACE_READ_ONLY",
            status_code=405,
            details={"workspace": key},
        )


class RemoteWorkspaceUnreachableError(MolExpError):
    """The transport to a remote workspace failed (connection / auth)."""

    def __init__(self, key: str, reason: str | None = None) -> None:
        super().__init__(
            message=f"Remote workspace '{key}' is unreachable" + (f": {reason}" if reason else ""),
            code="REMOTE_WORKSPACE_UNREACHABLE",
            status_code=502,
            details={"workspace": key, **({"reason": reason} if reason else {})},
        )
