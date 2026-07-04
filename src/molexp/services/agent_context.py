"""``build_mount_context`` — what a mounted agent session knows about its home.

The ONE mount-context builder shared by the server's session-create route and
``molexp agent chat`` (Python = UI law): given an optional scope
(project/experiment/run), it renders a compact, deterministic markdown block
from the canonical :class:`~molexp.workspace.workspace_context.WorkspaceContext`
read-model — the projection's first actual agent consumer (vision-loop-11).

Strict resolution: an id that does not resolve raises the workspace's typed
``*NotFoundError`` — never a silent downgrade to workspace scope. Snapshot
semantics: the block is built once at session start (and rebuilt verbatim from
the persisted scope on re-attach); the agent refreshes state through its read
tools, not through mid-conversation prompt mutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.workspace import ContextFocus, assemble_workspace_context

if TYPE_CHECKING:
    from molexp.workspace import Workspace
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run
    from molexp.workspace.workspace_context import WorkspaceContext

__all__ = ["build_mount_context", "mount_session_scope", "resolve_scope_dir"]

_MAX_LIST_ROWS = 10
_TRUNCATION_MARKER = "…(truncated)"
_SEP = "\n\n"


def build_mount_context(
    workspace: Workspace,
    *,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    max_chars: int = 4000,
) -> str:
    """Render the mounted entity's state as a system-prompt context block.

    Args:
        workspace: The live workspace the session is attached to.
        project_id: Optional project scope (required for deeper scopes).
        experiment_id: Optional experiment scope (requires *project_id*).
        run_id: Optional run scope (requires both parent ids).
        max_chars: Whole-block budget; over-budget content is cut at section
            boundaries with an explicit truncation marker (no silent caps).

    Returns:
        A deterministic markdown block for the session's system prompt.

    Raises:
        ProjectNotFoundError / ExperimentNotFoundError / RunNotFoundError:
            A named id does not resolve — never downgraded (no-fallback law).
        ValueError: A deeper id was given without its parents.
    """
    block, _ = _resolve_and_render(
        workspace,
        project_id=project_id,
        experiment_id=experiment_id,
        run_id=run_id,
        max_chars=max_chars,
    )
    return block


def resolve_scope_dir(
    workspace: Workspace,
    *,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Resolve the mount scope to its on-disk directory (session anchor).

    Same strict resolution as :func:`build_mount_context` — a bad id raises
    the typed ``*NotFoundError``. Returns ``None`` when no scope is given, so
    the session anchors at the workspace root as before.
    """
    project, experiment, run = _resolve_scope(
        workspace, project_id=project_id, experiment_id=experiment_id, run_id=run_id
    )
    deepest = run or experiment or project
    # Folder.path() returns molexp.path.Path, not a pathlib.Path subclass.
    return Path(str(deepest.path())) if deepest is not None else None


def mount_session_scope(
    workspace: Workspace,
    *,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    max_chars: int = 4000,
) -> tuple[str, Path | None]:
    """Resolve the scope ONCE and return ``(context_block, session_anchor)``.

    The composition CLI and server both consume (Python = UI): one strict
    resolution feeds both the rendered block and the on-disk anchor the
    session folder mounts at. No scope at all → ``("", None)`` — an unscoped
    session keeps its base prompt untouched (unlike :func:`build_mount_context`,
    whose no-scope case deliberately renders the workspace overview). A deeper
    id without its parents raises ``ValueError``; an unresolvable id raises the
    workspace-typed ``*NotFoundError``.
    """
    if not (project_id or experiment_id or run_id):
        return "", None
    return _resolve_and_render(
        workspace,
        project_id=project_id,
        experiment_id=experiment_id,
        run_id=run_id,
        max_chars=max_chars,
    )


def _resolve_and_render(
    workspace: Workspace,
    *,
    project_id: str | None,
    experiment_id: str | None,
    run_id: str | None,
    max_chars: int,
) -> tuple[str, Path | None]:
    """One strict resolution feeding both the rendered block and the anchor."""
    project, experiment, run = _resolve_scope(
        workspace, project_id=project_id, experiment_id=experiment_id, run_id=run_id
    )
    context = assemble_workspace_context(
        workspace,
        focus=ContextFocus(project_id=project_id, experiment_id=experiment_id, run_id=run_id),
    )
    if run is not None:
        sections = _run_sections(run, experiment_id or "", project_id or "")
    elif experiment is not None:
        sections = _experiment_sections(context, experiment_id or "", project_id or "")
    else:
        sections = _workspace_sections(context)
    deepest = run or experiment or project
    anchor = Path(str(deepest.path())) if deepest is not None else None
    return _fit(sections, max_chars), anchor


def _resolve_scope(
    workspace: Workspace,
    *,
    project_id: str | None,
    experiment_id: str | None,
    run_id: str | None,
) -> tuple[Project | None, Experiment | None, Run | None]:
    """Strictly resolve the (project, experiment, run) triple — never downgrade."""
    if run_id and not (project_id and experiment_id):
        raise ValueError("run_id requires project_id and experiment_id")
    if experiment_id and not project_id:
        raise ValueError("experiment_id requires project_id")
    project = workspace.get_project(project_id) if project_id else None
    experiment = project.get_experiment(experiment_id) if project and experiment_id else None
    run = experiment.get_run(run_id) if experiment and run_id else None
    return project, experiment, run


def _fit(sections: list[str], max_chars: int) -> str:
    """Join sections within budget, cutting at whole-section boundaries.

    The first section always survives (a mount block is never empty), so a
    single oversized header can exceed *max_chars*; every later section is
    dropped behind an explicit truncation marker once the budget is spent.
    """
    out: list[str] = []
    used = 0
    for section in sections:
        cost = len(section) + len(_SEP)
        if used + cost > max_chars - len(_TRUNCATION_MARKER) - len(_SEP) and out:
            out.append(_TRUNCATION_MARKER)
            break
        out.append(section)
        used += cost
    return _SEP.join(out)


def _run_sections(run: Run, experiment_id: str, project_id: str) -> list[str]:
    meta = run.metadata
    header = [
        f"# Mounted on run `{run.id}` ({project_id}/{experiment_id})",
        f"- status: {run.status}",
        f"- params: {dict(meta.parameters) if meta.parameters else '{}'}",
    ]
    if meta.config_hash:
        header.append(f"- config_hash: {meta.config_hash}")
    current = run.read_ops().current_execution_id
    if current:
        header.append(f"- execution: {current}")
    error = meta.error
    sections = ["\n".join(header)]
    if run.status == "failed" and error is not None:
        sections.append(f"## Error\n{error.type}: {error.message}")

    from molexp.workspace.assets import scan

    root = run.experiment.project.workspace.root
    artifacts = scan.scan_assets(root, producer_run=run.id, limit=_MAX_LIST_ROWS + 1)
    if artifacts:
        rows = [
            # Base ``Asset`` deliberately declares no ``kind`` — concrete
            # subclasses do (assets/base.py), hence the getattr.
            f"- {a.asset_id} · {getattr(a, 'kind', 'asset')} · {a.name}"
            for a in artifacts[:_MAX_LIST_ROWS]
        ]
        if len(artifacts) > _MAX_LIST_ROWS:
            rows.append(f"- (+{len(artifacts) - _MAX_LIST_ROWS} more)")
        sections.append("## Artifacts\n" + "\n".join(rows))
    return sections


def _experiment_sections(
    context: WorkspaceContext, experiment_id: str, project_id: str
) -> list[str]:
    exp_ref = next(
        (e for e in context.experiments if e.id == experiment_id),
        None,
    )
    header = [f"# Mounted on experiment `{experiment_id}` ({project_id})"]
    if exp_ref is not None and exp_ref.parameter_space:
        header.append(f"- parameter space: {dict(exp_ref.parameter_space)}")
    has_workflow = any(w.experiment_id == experiment_id for w in context.workflows)
    header.append(f"- workflow defined: {has_workflow}")

    runs = [
        r
        for r in (*context.recent_runs, *context.failed_runs, *context.running_runs)
        if r.experiment_id == experiment_id
    ]
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for ref in runs:
        if ref.run_id in seen:
            continue
        seen.add(ref.run_id)
        counts[ref.status] = counts.get(ref.status, 0) + 1
    if counts:
        header.append("- runs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    sections = ["\n".join(header)]
    rows = sorted({r.run_id: r for r in runs}.values(), key=lambda r: r.run_id)[:_MAX_LIST_ROWS]
    if rows:
        sections.append("## Recent runs\n" + "\n".join(f"- {r.run_id} · {r.status}" for r in rows))
    return sections


def _workspace_sections(context: WorkspaceContext) -> list[str]:
    ws_ref = context.workspace
    header = [
        f"# Mounted on workspace `{ws_ref.name}`",
        f"- projects: {len(context.projects)}, experiments: {len(context.experiments)}",
    ]
    sections = ["\n".join(header)]
    if context.projects:
        rows = [f"- {proj.id}" for proj in context.projects[:_MAX_LIST_ROWS]]
        if len(context.projects) > _MAX_LIST_ROWS:
            rows.append(f"- (+{len(context.projects) - _MAX_LIST_ROWS} more)")
        sections.append("## Projects\n" + "\n".join(rows))
    if context.failed_runs or context.running_runs:
        rows = [
            f"- {r.run_id} · {r.status}"
            for r in (*context.running_runs, *context.failed_runs)[:_MAX_LIST_ROWS]
        ]
        sections.append("## Live / failed runs\n" + "\n".join(rows))
    if context.knowledge:
        rows = [f"- {k.title} ({k.path})" for k in context.knowledge[:_MAX_LIST_ROWS]]
        sections.append("## Knowledge\n" + "\n".join(rows))
    if context.stale_or_missing:
        rows = [
            f"- [{f.kind}] {f.ref}: {f.detail}" for f in context.stale_or_missing[:_MAX_LIST_ROWS]
        ]
        sections.append("## Health flags\n" + "\n".join(rows))
    return sections
