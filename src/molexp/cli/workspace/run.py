"""``molexp {run,exec,shell}`` — execution commands."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from molexp._typing import JSONValue
from molexp.cli._app import app
from molexp.cli._common import console, deterministic_run_id, reap_zombie_run, rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.profile import MolCfg, ProfileConfig, load_molcfg
from molexp.profile.loader import find_default_config
from molexp.workflow import WorkflowRuntime, default_binding_registry
from molexp.workspace.run import RETRYABLE_STATUSES
from molexp.workspace.target import LocalTarget, RemoteTarget

if TYPE_CHECKING:
    from molexp.workflow.protocols import RunContextLike
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run
    from molexp.workspace.workspace import Workspace

RunHandler = Callable[["Path", "Run", "Experiment", "Project"], None]

# (existing_run, run_params, run_id, seed_label) — one prospective run of an
# experiment, before the verb-specific skip/create rules are applied.
RunCandidate = tuple["Run | None", dict[str, JSONValue], str, "str | None"]

_SCRIPT_ARG = Annotated[
    Path,
    typer.Argument(
        help="Python script with me.entry(project) call.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]


# ── Config helpers ────────────────────────────────────────────────────────────


def _coerce_value(raw: str) -> JSONValue:
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(d: dict[str, JSONValue], key_path: str, value: JSONValue) -> None:
    parts = key_path.split(".")
    node: dict[str, JSONValue] = d
    for part in parts[:-1]:
        existing = node.get(part)
        if isinstance(existing, dict):
            node = existing
            continue
        new_child: dict[str, JSONValue] = {}
        node[part] = new_child
        node = new_child
    node[parts[-1]] = value


def _apply_overrides(profile_cfg: ProfileConfig, overrides: list[str]) -> ProfileConfig:
    if not overrides:
        return profile_cfg
    data = profile_cfg.to_dict()
    for item in overrides:
        if "=" not in item:
            rprint(f"[red]Error:[/red] --set value {item!r} is not in KEY=VALUE format.")
            raise typer.Exit(1)
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            rprint(f"[red]Error:[/red] --set value {item!r} has an empty key.")
            raise typer.Exit(1)
        _set_nested(data, key, _coerce_value(raw))
    return ProfileConfig(data, name=profile_cfg.name)


def _resolve_profile(config_path: Path | None, profile: str | None) -> ProfileConfig:
    resolved_path: Path | None
    if config_path is not None:
        resolved_path = config_path
    else:
        resolved_path = find_default_config()
    if resolved_path is None:
        if profile is not None:
            rprint(
                f"[red]Error:[/red] --profile {profile!r} was requested but no config file found."
            )
            raise typer.Exit(1)
        return ProfileConfig({}, name=None)
    try:
        cfg: MolCfg = load_molcfg(resolved_path)
    except Exception as exc:
        rprint(f"[red]Error:[/red] failed to load config {resolved_path}: {exc}")
        raise typer.Exit(1)  # noqa: B904
    try:
        return cfg.resolve(profile)
    except KeyError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)  # noqa: B904


# ── Replica dispatch ──────────────────────────────────────────────────────────


def _watch_path_for(workspace_arg: Path | None, submitted: list[Run]) -> str:
    root: Path | None = None
    if workspace_arg is not None:
        root = Path(workspace_arg)
    elif submitted:
        try:
            root = Path(submitted[0].experiment.project.workspace.root)
        except AttributeError:
            root = None
    if root is None:
        return "."
    root = root.resolve()
    cwd = Path.cwd().resolve()
    if root == cwd:
        return "."
    try:
        return str(root.relative_to(cwd))
    except ValueError:
        return str(root)


def _load_script_workspaces(
    script: Path, workspace: Path | None, explicit_workspace: bool
) -> tuple[list[Workspace], Path | None]:
    """Import *script* and collect its ``me.entry()`` workspaces.

    Root precedence: an explicit -ws flag is a STRONG override (wins over a
    script-hardcoded root); otherwise infer the entry-script directory as a
    WEAK override (only fills a rootless Workspace), falling back to cwd.
    """
    from molexp.entry import infer_workspace_root, load_workspaces
    from molexp.workspace.workspace import set_cli_root_override

    if explicit_workspace and workspace is not None:
        override_path = Path(workspace).resolve()
        set_cli_root_override(override_path, explicit=True)
    else:
        try:
            override_path = infer_workspace_root(script)
        except ValueError:
            override_path = Path.cwd().resolve()
        set_cli_root_override(override_path, explicit=False)
    try:
        workspaces = load_workspaces(script)
    except Exception as exc:
        rprint(f"[red]Error importing {script.name}:[/red] {exc}")
        rprint(traceback.format_exc(), end="")
        raise typer.Exit(1)  # noqa: B904
    finally:
        set_cli_root_override(None)

    if not workspaces:
        rprint(
            "[red]Error:[/red] No me.entry() call found in script. Add [bold]me.entry(workspace)[/bold] at module level."
        )
        raise typer.Exit(1)
    return workspaces, override_path


def _experiment_candidates(
    exp: Experiment, profile_cfg: ProfileConfig
) -> tuple[list[RunCandidate], bool, int]:
    """Build the prospective run list for one experiment.

    Returns ``(candidates, use_declared_runs, total)``.
    """
    existing_runs = sorted(exp.list_runs(), key=lambda item: item.id)
    # A run auto-generated by the replica path carries a "replica"
    # marker in its parameters; only genuinely user-declared runs
    # count as "declared". Without this, a prior replica invocation
    # would make a later profile reuse that run instead of creating
    # its own profile-distinct replica run.
    declared_runs = [r for r in existing_runs if "replica" not in r.parameters]
    use_declared_runs = len(declared_runs) > 0
    existing_by_id = {r.id: r for r in existing_runs}
    seeds = exp.get_seeds()
    total = len(declared_runs) if use_declared_runs else exp.n_replicas

    candidates: list[RunCandidate]
    if use_declared_runs:
        candidates = [(run, dict(run.parameters), run.id, None) for run in declared_runs]
    else:
        candidates = []
        for replica_idx, seed in enumerate(seeds):
            run_params = {**exp.params, "seed": seed, "replica": replica_idx}
            id_seed = dict(run_params)
            if profile_cfg.name is not None:
                id_seed["_profile"] = profile_cfg.name
                id_seed["_config_hash"] = profile_cfg.content_hash()
            run_id = deterministic_run_id(id_seed)
            # Bind to an existing run with the same profile-aware id so
            # the shared skip/resume/profile checks below apply; a
            # different profile derives a new id and creates its own run.
            candidates.append((existing_by_id.get(run_id), run_params, run_id, str(seed)))
    return candidates, use_declared_runs, total


def _select_candidate_runs(
    candidates: list[RunCandidate],
    *,
    exp: Experiment,
    continue_verb: str | None,
    profile_cfg: ProfileConfig,
) -> list[tuple[Run, str]]:
    """Apply the verb-specific skip/create rules and report each decision."""
    selected_runs: list[tuple[Run, str]] = []
    for existing, run_params, run_id, seed_label in candidates:
        mol_run = existing
        if mol_run is not None and mol_run.status == "running" and reap_zombie_run(mol_run):
            rprint(
                f"  [yellow]![/yellow] {exp.id}  run={mol_run.id} (stale 'running' run reaped -> failed)"
            )
        if continue_verb is not None:
            # resume / rerun own exactly the finished-but-not-
            # succeeded runs (failed / cancelled). pending is plain
            # run's job, succeeded is done, and a live running run
            # must never get a second execution — all skipped, which
            # keeps the three verbs orthogonal. The retryable domain
            # is the shared workspace RETRYABLE_STATUSES.
            if mol_run is None:
                rprint(f"  [dim]- {exp.id}  run={run_id} (no existing run, skipped)[/dim]")
                continue
            if mol_run.status not in RETRYABLE_STATUSES:
                rprint(
                    f"  [dim]- {exp.id}  run={mol_run.id} ({mol_run.status}, skipped — "
                    f"{continue_verb} only retries failed/cancelled runs)[/dim]"
                )
                continue
            if mol_run.metadata.profile != profile_cfg.name:
                rprint(f"  [dim]- {exp.id}  run={mol_run.id} (profile mismatch, skipped)[/dim]")
                continue
        elif mol_run is not None:
            # plain run: run only what has not run yet (pending).
            # Leave succeeded / running / failed / cancelled alone —
            # retrying a failure is an explicit --resume / --rerun.
            if mol_run.status != "pending":
                rprint(
                    f"  [dim]- {exp.id}  run={mol_run.id} ({mol_run.status}, skipped — "
                    "use --resume or --rerun to retry)[/dim]"
                )
                continue
        else:
            mol_run = exp.add_run(params=run_params, id=run_id)
        label_text = f"seed={seed_label}" if seed_label is not None else f"run={mol_run.id}"
        selected_runs.append((mol_run, label_text))
        icon = "[cyan]>[/cyan]" if continue_verb is not None else "[dim]o[/dim]"
        rprint(f"  {icon} {exp.id}  {label_text}")
    return selected_runs


def _execute_selected(
    all_replicas: list[tuple[Run, Experiment, Project]],
    run_handler: RunHandler,
    script: Path,
    *,
    show_progress: bool,
) -> list[Run]:
    """Run *run_handler* over every selected run, with optional progress bar."""
    dispatched_runs: list[Run] = []
    if show_progress and all_replicas:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("running workflows", total=len(all_replicas))
            for mol_run, exp, project in all_replicas:
                progress.update(task_id, description=f"{exp.id} / {mol_run.id}")
                run_handler(script, mol_run, exp, project)
                dispatched_runs.append(mol_run)
                progress.advance(task_id)
    else:
        for mol_run, exp, project in all_replicas:
            run_handler(script, mol_run, exp, project)
            dispatched_runs.append(mol_run)
            rprint(f"  [cyan]>[/cyan] dispatched {exp.id}  run={mol_run.id}")
    return dispatched_runs


def _dispatch_runs(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    continue_verb: str | None,
    workspace: Path | None,
    explicit_workspace: bool,
    run_handler: RunHandler | None,
    mode_label: str,
    suppress_ok: bool = False,
    dry_run: bool = False,
    show_progress: bool = False,
) -> tuple[int, list[Run]]:
    workspaces, override_path = _load_script_workspaces(script, workspace, explicit_workspace)

    total_dispatched = 0
    all_replicas: list[tuple[Run, Experiment, Project]] = []

    for ws in workspaces:
        if override_path is not None and ws.root == override_path:
            rprint(f"[dim]--workspace override active: {ws.root}[/dim]")
        for project in ws.list_projects():
            for exp in project.list_experiments():
                if default_binding_registry.for_experiment(exp) is None:
                    continue
                candidates, use_declared_runs, total = _experiment_candidates(exp, profile_cfg)
                label = (
                    f"[cyan]{continue_verb}[/cyan] + {mode_label}"
                    if continue_verb is not None
                    else mode_label
                )
                profile_display = profile_cfg.name or "(defaults)"
                rprint(
                    f"\n[bold]Experiment:[/bold] {exp.name}"
                    f"\n  Script:    {script}"
                    f"\n  Workspace: {ws.root}"
                    f"\n  Project:   {project.name}"
                    f"\n  Profile:   {profile_display}"
                    f"\n  Runs:      {total} {'declared' if use_declared_runs else 'replicas'}"
                    f"\n  Mode:      {label}"
                )
                selected_runs = _select_candidate_runs(
                    candidates, exp=exp, continue_verb=continue_verb, profile_cfg=profile_cfg
                )

                submit_cwd_str = str(Path.cwd().resolve())
                for mol_run, _label_text in selected_runs:
                    if not dry_run:
                        mol_run._update_metadata(
                            script=str(script.resolve()),
                            submit_cwd=submit_cwd_str,
                            profile=profile_cfg.name,
                            config=profile_cfg.to_dict(),
                            config_hash=(
                                profile_cfg.content_hash()
                                if len(profile_cfg) > 0 or profile_cfg.name
                                else None
                            ),
                        )
                    all_replicas.append((mol_run, exp, project))
                total_dispatched += len(selected_runs)

    if dry_run or run_handler is None:
        if not suppress_ok:
            rprint(f"\n[green]OK[/green] compiled workflow plan: {total_dispatched} run(s) ready.")
        return total_dispatched, [mol_run for mol_run, _exp, _project in all_replicas]

    dispatched_runs = _execute_selected(
        all_replicas, run_handler, script, show_progress=show_progress
    )
    if not suppress_ok:
        verb = {"resume": "resumed", "rerun": "reran"}.get(continue_verb or "", "completed")
        rprint(f"\n[green]OK[/green] {total_dispatched} runs {verb}.")
    return total_dispatched, dispatched_runs


def _make_local_inprocess_handler(
    profile_cfg: ProfileConfig, *, verb: str | None = None
) -> RunHandler:
    import asyncio

    from molexp.workflow import seed_from_execution
    from molexp.workspace.run import RunContext

    def _handler(_script: Path, mol_run: Run, experiment: Experiment, _project: Project) -> None:
        spec = default_binding_registry.for_experiment(experiment)
        if spec is None:
            raise RuntimeError(f"Experiment {experiment.name!r} has no workflow attached.")
        execution_id: str | None = None
        seed_outputs = None
        if verb == "resume":
            # Reopen the last failed/interrupted execution and seed its
            # completed nodes; the no-fallback semantics live with the
            # workflow layer (see ``seed_from_execution``).
            execution_id, seed_outputs = seed_from_execution(mol_run)
        with RunContext(mol_run, profile_config=profile_cfg, execution_id=execution_id) as ctx:
            asyncio.run(
                WorkflowRuntime().execute(
                    spec,
                    run_context=cast("RunContextLike", ctx),
                    execution_id=execution_id,
                    seed_outputs=seed_outputs,
                )
            )

    return _handler


@app.command(name="execute")
def execute(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory to execute (…/runs/run-<id>).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
            help="Execution attempt to (re)open; already-completed nodes are seeded.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to molcfg file.", exists=True),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile name to apply."),
    ] = None,
) -> None:
    """Worker entry point — execute one Run's workflow in this process.

    This is what the molq submit plugin launches on the scheduler:
    ``python -m molexp.cli execute <run_dir> --execution-id <eid>``. molq itself
    only runs whatever argv it is handed; reconstructing the workflow is the
    application's job, and that is this command. It reads the run's recorded
    defining ``script``, imports it to rebuild the experiment→workflow binding
    (the same registry ``molexp run`` populates), locates the Run, and executes
    its workflow against *execution_id* — reopening that attempt and seeding any
    already-completed nodes so a resubmitted job continues rather than restarts.
    """
    import asyncio

    from molexp._run_display import read_run_json
    from molexp.entry import load_workspaces
    from molexp.workflow import read_node_outputs
    from molexp.workspace import (
        ExperimentNotFoundError,
        ProjectNotFoundError,
        RunNotFoundError,
    )
    from molexp.workspace.run import RunContext

    run_dir = Path(run_dir)
    meta = read_run_json(run_dir)
    if not meta:
        rprint(f"[red]Error:[/red] no readable run.json under {run_dir}")
        raise typer.Exit(1)
    ctx_meta = meta.get("context") or {}
    run_id = meta.get("id") or ctx_meta.get("run_id") or run_dir.name.removeprefix("run-")
    # project/experiment ids are not stored in run.json; derive them from the
    # canonical layout …/projects/<P>/experiments/<E>/runs/run-<id>. The run
    # context only carries them once a run has executed, so a fresh run relies on
    # the path; fall back to context if the layout ever differs.
    project_id = ctx_meta.get("project_id")
    experiment_id = ctx_meta.get("experiment_id")
    try:
        experiment_id = experiment_id or run_dir.parent.parent.name
        project_id = project_id or run_dir.parent.parent.parent.parent.name
    except (IndexError, AttributeError):
        pass
    script = meta.get("script")
    if not script:
        rprint(
            "[red]Error:[/red] run.json has no 'script' field; a worker cannot "
            "rebuild the workflow without the defining script. Submit via "
            "`molexp run <script> --scheduler …` so the path is recorded."
        )
        raise typer.Exit(1)
    if not (run_id and project_id and experiment_id):
        rprint(f"[red]Error:[/red] run.json under {run_dir} is missing project/experiment/run ids.")
        raise typer.Exit(1)

    # Import the defining script in this fresh process: its module-level
    # ``me.entry()`` + ``exp.run()`` register the workspaces and the
    # experiment→workflow binding the runtime resolves below.
    workspaces = load_workspaces(Path(script))
    run_obj: Run | None = None
    experiment: Experiment | None = None
    for ws in workspaces:
        try:
            run_obj = ws.get_project(project_id).get_experiment(experiment_id).get_run(run_id)
            experiment = ws.get_project(project_id).get_experiment(experiment_id)
            break
        except (ProjectNotFoundError, ExperimentNotFoundError, RunNotFoundError):
            continue
    if run_obj is None or experiment is None:
        rprint(
            f"[red]Error:[/red] {script} did not define run "
            f"{project_id}/{experiment_id}/{run_id}."
        )
        raise typer.Exit(1)

    spec = default_binding_registry.for_experiment(experiment)
    if spec is None:
        rprint(f"[red]Error:[/red] experiment {experiment_id!r} has no workflow attached.")
        raise typer.Exit(1)

    seed_outputs = read_node_outputs(run_obj.run_dir, execution_id) if execution_id else None
    # Seed the profile config with the run's persisted config + its own dir so
    # the workflow can locate ``work_dir`` (the in-process ``molexp run`` path
    # seeds this via RunContext). Any --config/--profile overlays the persisted
    # values; the run dir is authoritative and always wins.
    overlay = _resolve_profile(config, profile)
    merged: dict[str, JSONValue] = dict(run_obj.metadata.config or {})
    merged.update(overlay.to_dict())
    merged["work_dir"] = str(run_obj.run_dir)
    profile_cfg = ProfileConfig(merged, name=overlay.name or run_obj.metadata.profile)
    rprint(f"[dim]execute[/dim] run={run_id} execution={execution_id or '(new)'}")
    with RunContext(run_obj, profile_config=profile_cfg, execution_id=execution_id) as ctx:
        asyncio.run(
            WorkflowRuntime().execute(
                spec,
                run_context=cast("RunContextLike", ctx),
                execution_id=execution_id,
                seed_outputs=seed_outputs,
            )
        )
    rprint(f"[green]OK[/green] execute complete run={run_id} status={run_obj.status}")


def _spawn_background_local_run(
    *,
    script: Path,
    target_path: Path,
    config: Path | None,
    profile: str | None,
    overrides: list[str],
    resume: bool,
    rerun: bool,
) -> None:
    """Spawn a detached local ``molexp run`` worker for long-running jobs."""
    log_dir = target_path / ".molexp" / "background"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{script.stem}-{stamp}.log"
    cmd = [
        sys.executable,
        "-c",
        "from molexp.cli import app; app()",
        "run",
        str(script),
        "--local",
        "-t",
        str(target_path),
    ]
    if config is not None:
        cmd.extend(["--config", str(config)])
    if profile is not None:
        cmd.extend(["--profile", profile])
    for item in overrides:
        cmd.extend(["--override", item])
    if resume:
        cmd.append("--resume")
    if rerun:
        cmd.append("--rerun")

    with log_path.open("ab") as log:
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    rprint(f"[green]OK[/green] background run started pid={process.pid}")
    rprint(f"[dim]Log: {log_path}[/dim]")


def _select_backend(
    *,
    target_path: Path,
    local: bool,
    scheduler: str | None,
    target_cli: str | None,
) -> tuple[ComputeTarget | None, str | None, bool]:
    """Resolve the execution backend from the mutually exclusive flags.

    Returns ``(selected_target, selected_scheduler, is_local)``.
    """
    backend_flags = sum(1 for f in (local, scheduler is not None, target_cli is not None) if f)
    if backend_flags > 1:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag (--local, --scheduler, --target)."
        )
        raise typer.Exit(1)

    selected_target = None
    if target_cli is not None:
        from molexp.workspace import Workspace, get_target

        ws = Workspace(target_path)
        try:
            selected_target = get_target(ws, target_cli)
        except KeyError as exc:
            rprint(f"[red]{exc}[/red] — see `molexp target list`.")
            raise typer.Exit(1) from exc

    selected_scheduler = selected_target.scheduler if selected_target is not None else scheduler
    is_local = selected_scheduler is None and selected_target is None
    return selected_target, selected_scheduler, is_local


def _profile_mode_label(profile_cfg: ProfileConfig, backend: str) -> str:
    profile_label = (
        f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
    )
    return f"{backend} profile={profile_label}"


def _run_dry_run(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    continue_verb: str | None,
    target_path: Path,
    explicit_ws: bool,
) -> None:
    """Compile/materialize the run plan without executing tasks."""
    mode_label = "[blue]dry-run[/blue] compile-only"
    n, runs = _dispatch_runs(
        script=script,
        profile_cfg=profile_cfg,
        continue_verb=continue_verb,
        workspace=target_path,
        explicit_workspace=explicit_ws,
        run_handler=None,
        mode_label=mode_label,
        dry_run=True,
    )
    if runs:
        watch_arg = _watch_path_for(target_path, runs)
        rprint(f"[dim]Preview with: molexp serve -t {watch_arg}[/dim]")
    elif n == 0:
        rprint("[dim]No runnable bound experiments found.[/dim]")


def _run_local_inprocess(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    continue_verb: str | None,
    target_path: Path,
    explicit_ws: bool,
) -> None:
    """Execute the run plan in-process, with a progress bar."""
    mode_label = _profile_mode_label(profile_cfg, "[green]local[/green]")
    _dispatch_runs(
        script=script,
        profile_cfg=profile_cfg,
        continue_verb=continue_verb,
        workspace=target_path,
        explicit_workspace=explicit_ws,
        run_handler=_make_local_inprocess_handler(profile_cfg, verb=continue_verb),
        mode_label=mode_label,
        show_progress=True,
    )


def _submit_to_scheduler(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    continue_verb: str | None,
    target_path: Path,
    explicit_ws: bool,
    selected_target: ComputeTarget | None,
    selected_scheduler: str | None,
    cluster: str | None,
    resources: dict[str, JSONValue],
    scheduling: dict[str, JSONValue],
    block: bool,
) -> None:
    """Submit the run plan through molq and report (or monitor) the result."""
    from molexp.plugins.submit_molq.metadata import supported_schedulers
    from molexp.plugins.submit_molq.submit import make_submit_handler

    available_schedulers = supported_schedulers()
    if available_schedulers and selected_scheduler not in available_schedulers:
        supported_text = ", ".join(available_schedulers)
        rprint(
            f"[red]Error:[/red] Unsupported molq scheduler: {selected_scheduler!r}. Available: {supported_text}"
        )
        raise typer.Exit(1)

    assert selected_scheduler is not None
    handler = make_submit_handler(
        scheduler=selected_scheduler,
        cluster=cluster,
        resources=resources,
        scheduling=scheduling,
        target=selected_target,
    )
    mode_label = _profile_mode_label(profile_cfg, f"[magenta]{selected_scheduler}[/magenta]")

    n, submitted = _dispatch_runs(
        script=script,
        profile_cfg=profile_cfg,
        continue_verb=continue_verb,
        workspace=target_path,
        explicit_workspace=explicit_ws,
        run_handler=handler,
        mode_label=mode_label,
        suppress_ok=True,
    )
    if n == 0:
        verb = {"resume": "resumed", "rerun": "reran"}.get(continue_verb or "", "submitted")
        rprint(f"\n[dim]No runs {verb}.[/dim]")
        return

    watch_arg = _watch_path_for(target_path, submitted)
    if block and submitted:
        from molexp.cli.tui import RunMonitor

        rprint(f"\n[dim]Submitted {n} runs. Opening monitor… (press q to close)[/dim]")
        RunMonitor(title=f"{script.stem}  [{mode_label}]").watch(submitted)
        rprint(f"\n[dim]Monitor closed. {n} runs are still executing (if any).[/dim]")
        rprint(f"[dim]Reopen with:  molexp monitor -t {watch_arg}[/dim]")
    else:
        verb = {"resume": "resumed", "rerun": "reran"}.get(continue_verb or "", "submitted")
        rprint(f"\n[green]OK[/green] {n} runs {verb}.")
        if submitted:
            rprint(f"[dim]Monitor runs with:  molexp monitor -t {watch_arg}[/dim]")


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command()
def run(
    script: _SCRIPT_ARG,
    local: Annotated[
        bool,
        typer.Option(
            "--local", help="Run in-process, no scheduler.", rich_help_panel="Execution Backend"
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Compile/materialize the workspace and run plan without executing tasks.",
        ),
    ] = False,
    bg: Annotated[
        bool,
        typer.Option(
            "--bg",
            help="Run local workflows in a detached background process.",
            rich_help_panel="Execution Backend",
        ),
    ] = False,
    scheduler: Annotated[
        str | None,
        typer.Option(
            "--scheduler",
            help="molq scheduler backend (local, slurm, pbs, lsf).",
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to molcfg file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    profile: Annotated[str | None, typer.Option("--profile", help="Named molcfg profile.")] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--override", help="Override config key (KEY=VALUE, repeatable)."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Reopen each non-succeeded run's last execution and continue at "
                "workflow-node granularity (seed already-completed nodes from disk, "
                "recompute the rest). Mutually exclusive with --rerun."
            ),
        ),
    ] = False,
    rerun: Annotated[
        bool,
        typer.Option(
            "--rerun",
            help="Re-execute non-succeeded runs from scratch in a new execution (no seed).",
        ),
    ] = False,
    cpus: Annotated[
        int | None, typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="HPC Options")
    ] = None,
    mem: Annotated[
        str | None,
        typer.Option("--mem", help="Memory per job (e.g. 8G).", rich_help_panel="HPC Options"),
    ] = None,
    time: Annotated[
        str | None,
        typer.Option("--time", help="Wall-clock time limit.", rich_help_panel="HPC Options"),
    ] = None,
    gpus: Annotated[
        int | None, typer.Option("--gpus", help="GPUs per job.", rich_help_panel="HPC Options")
    ] = None,
    gpu_type: Annotated[
        str | None,
        typer.Option(
            "--gpu-type", help="GPU type constraint (e.g. a100).", rich_help_panel="HPC Options"
        ),
    ] = None,
    account: Annotated[
        str | None,
        typer.Option(
            "--account", "-A", help="Account / project name.", rich_help_panel="HPC Options"
        ),
    ] = None,
    cluster: Annotated[
        str | None,
        typer.Option("--cluster", help="molq cluster name.", rich_help_panel="HPC Options"),
    ] = None,
    target_cli: Annotated[
        str | None,
        typer.Option(
            "--compute-target",
            help="Named compute target from workspace.",
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    partition: Annotated[
        str | None,
        typer.Option("--partition", "-p", help="SLURM partition.", rich_help_panel="SLURM Options"),
    ] = None,
    qos: Annotated[
        str | None, typer.Option("--qos", help="SLURM QOS.", rich_help_panel="SLURM Options")
    ] = None,
    queue: Annotated[
        str | None,
        typer.Option(
            "--queue", "-q", help="PBS/LSF queue name.", rich_help_panel="PBS / LSF Options"
        ),
    ] = None,
    block: Annotated[
        bool,
        typer.Option(
            "--block", help="Block and open monitor after submit.", rich_help_panel="HPC Options"
        ),
    ] = False,
    target_spec: TargetOption = ".",
) -> None:
    """Execute the workflow(s) defined by *script*."""
    # An explicit -ws/--workspace flag (anything other than the "." cwd default)
    # is a strong root override; absence means infer the script's directory.
    explicit_ws = target_spec != "."
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        rprint("[red]Error:[/red] 'run' on remote targets is not yet supported.")
        rprint("  Use [bold]molexp exec[/bold] or [bold]shell[/bold] for remote execution.")
        raise typer.Exit(1)

    if dry_run and bg:
        rprint("[red]Error:[/red] --dry-run and --bg cannot be combined.")
        raise typer.Exit(1)

    if resume and rerun:
        rprint("[red]Error:[/red] --resume and --rerun are mutually exclusive.")
        raise typer.Exit(1)
    continue_verb = "resume" if resume else ("rerun" if rerun else None)

    selected_target, selected_scheduler, is_local = _select_backend(
        target_path=target.path,
        local=local,
        scheduler=scheduler,
        target_cli=target_cli,
    )

    profile_cfg = _resolve_profile(config, profile)
    profile_cfg = _apply_overrides(profile_cfg, overrides or [])

    if bg:
        if not is_local:
            rprint("[red]Error:[/red] --bg is only supported for local execution.")
            raise typer.Exit(1)
        _spawn_background_local_run(
            script=script,
            target_path=Path(target.path),
            config=config,
            profile=profile,
            overrides=overrides or [],
            resume=resume,
            rerun=rerun,
        )
        return

    if dry_run:
        _run_dry_run(
            script=script,
            profile_cfg=profile_cfg,
            continue_verb=continue_verb,
            target_path=target.path,
            explicit_ws=explicit_ws,
        )
        return

    if is_local:
        _run_local_inprocess(
            script=script,
            profile_cfg=profile_cfg,
            continue_verb=continue_verb,
            target_path=target.path,
            explicit_ws=explicit_ws,
        )
        return

    selected_queue = partition if partition is not None else queue
    _submit_to_scheduler(
        script=script,
        profile_cfg=profile_cfg,
        continue_verb=continue_verb,
        target_path=target.path,
        explicit_ws=explicit_ws,
        selected_target=selected_target,
        selected_scheduler=selected_scheduler,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "gpu_type": gpu_type, "time": time},
        scheduling={"queue": selected_queue, "account": account, "qos": qos},
        block=block,
    )


@app.command(name="exec")
def exec_cmd(
    command: Annotated[
        list[str] | None, typer.Argument(help="Command to execute on the target")
    ] = None,
    cwd: Annotated[
        str | None, typer.Option("--cwd", help="Working directory on the target")
    ] = None,
    timeout: Annotated[float | None, typer.Option("--timeout", help="Timeout in seconds")] = None,
    target_spec: TargetOption = ".",
) -> None:
    """Execute a command on the workspace target (local or remote transport)."""
    target, transport, _fs = resolve_workspace_target(target_spec)

    cmd: list[str] = list(command) if command else []
    if not cmd:
        rprint("[red]Error:[/red] No command provided.")
        raise typer.Exit(1)

    if isinstance(target, RemoteTarget):
        remote_cwd = cwd or target.path
    else:
        remote_cwd = cwd or str(target.path)

    rprint(f"[dim]{'[' + str(target) + ']'} {' '.join(cmd)}[/dim]")
    try:
        result = transport.run(cmd, cwd=remote_cwd, timeout=timeout)
    except Exception as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if result.stdout:
        rprint(result.stdout, end="")
    if result.stderr:
        from rich import print as _rprint

        _rprint(f"[yellow]{result.stderr}[/yellow]", end="")
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def shell(target_spec: TargetOption = ".") -> None:
    """Open an interactive shell on the target (SSH for remote, $SHELL for local)."""
    target, _transport, _fs = resolve_workspace_target(target_spec)

    if isinstance(target, RemoteTarget):
        import subprocess

        user_host = f"{target.user}@{target.host}" if target.user else target.host
        ssh_cmd = ["ssh", "-t", user_host, f"cd {target.path} && exec ${{SHELL:-bash}}"]
        if target.port:
            ssh_cmd.insert(1, "-p")
            ssh_cmd.insert(2, str(target.port))
        rprint(f"[dim]Opening shell to {user_host}...[/dim]")
        subprocess.run(ssh_cmd)
    else:
        import subprocess

        shell_bin = os.environ.get("SHELL", "bash")
        rprint(f"[dim]Opening shell in {target.path}...[/dim]")
        subprocess.run([shell_bin], cwd=str(target.path))
