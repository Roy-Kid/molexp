"""``resolve_plan_compute_target`` — the ONE plan compute-target resolution path.

CLI (`molexp plan`) and server (plan-tasks route) resolve the descriptive
compute target for PlanMode's step-9 execution report through this single
function (Python = UI law). The target is *descriptive*: it feeds
``GenerateExecutionReport`` and never schedules anything — server-side
``execute=true`` still runs the driver as an executor subprocess of the
serving host, exactly like the CLI does on its host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.workspace import Workspace
    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.run import Run

__all__ = ["resolve_plan_compute_target"]


def resolve_plan_compute_target(
    run: Run,
    workspace: Workspace,
    *,
    name: str | None = None,
) -> ComputeTarget | None:
    """Resolve the ``ComputeTarget`` the plan's execution report describes.

    Resolution order:

    1. An explicit *name* wins, resolved through the workspace's canonical
       :func:`~molexp.workspace.targets.resolve_compute_target` (the single
       named-target resolution path — the built-in ``local`` target resolves
       like any registered name). An unknown name raises ``ValueError``
       listing the advertised (effective) target names — never a silent
       fallback.
    2. Otherwise the run's ``metadata.target`` (stamped at run creation) is
       looked up in ``WorkspaceMetadata.targets``.
    3. Neither names a target → ``None`` (the documented local default).

    Args:
        run: The plan's workspace Run (its metadata may name a target).
        workspace: The workspace whose target registry is consulted.
        name: Optional explicit target name (the route's ``compute_target``).

    Returns:
        The resolved target, or ``None`` for the local default.

    Raises:
        ValueError: *name* is given but matches no advertised target.
    """
    if name:
        from molexp.workspace.targets import effective_targets, resolve_compute_target

        try:
            return resolve_compute_target(workspace, name)
        except KeyError as exc:
            known = sorted(t.name for t in effective_targets(workspace))
            raise ValueError(
                f"unknown compute target {name!r}; known targets: {known or '(none registered)'}"
            ) from exc
    targets = getattr(getattr(workspace, "metadata", None), "targets", []) or []
    target_name = getattr(run.metadata, "target", None)
    if not target_name:
        return None
    return next((t for t in targets if t.name == target_name), None)
