"""``molexp run`` — execute workflows defined by a Python script."""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import typer

from molexp.config import MolCfg, ProfileConfig, load_molcfg
from molexp.config.loader import find_default_config

from . import app
from ._common import (
    console,
    deterministic_run_id,
    reap_zombie_run,
    rprint,
)

RunHandler = Callable[[Any, Any, Any, Any], None]


# ── Config / profile resolution ─────────────────────────────────────────────


def _coerce_value(raw: str) -> Any:
    """Coerce a string CLI value to int, float, bool, or str."""
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


def _set_nested(d: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested dict value using dot-notation key path (mutates *d*)."""
    parts = key_path.split(".")
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _apply_overrides(
    profile_cfg: ProfileConfig,
    overrides: list[str],
) -> ProfileConfig:
    """Return a new :class:`ProfileConfig` with *overrides* applied on top.

    Each entry in *overrides* must be a ``KEY=VALUE`` string.  Dot notation
    is supported for nested keys (e.g. ``model.n_layers=3``).  Values are
    coerced to ``bool``, ``int``, ``float``, or ``str`` in that priority.

    Raises :class:`typer.Exit` with an error message on malformed entries.
    """
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
    """Load molcfg and resolve the requested profile.

    - If *config_path* is given it must exist.
    - Otherwise we look for a default ``molcfg.yaml`` / ``.yml`` / ``.json``
      in the current working directory.
    - If no file is found and no *profile* is requested, return an
      empty :class:`ProfileConfig` (defaults-only, name=None).
    - If no file is found but *profile* is requested, abort.
    """
    resolved_path: Path | None
    if config_path is not None:
        resolved_path = config_path
    else:
        resolved_path = find_default_config()

    if resolved_path is None:
        if profile is not None:
            rprint(
                f"[red]Error:[/red] --profile {profile!r} was requested but "
                "no config file was found (searched for ./molcfg.yaml / .yml / .json)."
            )
            raise typer.Exit(1)
        return ProfileConfig({}, name=None)

    try:
        cfg: MolCfg = load_molcfg(resolved_path)
    except Exception as exc:
        rprint(f"[red]Error:[/red] failed to load config {resolved_path}: {exc}")
        raise typer.Exit(1)

    try:
        return cfg.resolve(profile)
    except KeyError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


# ── Sweep driver ────────────────────────────────────────────────────────────


def _resolve_jobs(cli_jobs: int | None, profile_cfg: ProfileConfig) -> int:
    """Resolve effective sweep concurrency.

    CLI ``-j`` wins when provided; otherwise fall back to the profile's
    ``jobs:`` field; default to ``1`` (serial, backwards compatible).
    Values ``<= 0`` are clamped to ``1`` (``Semaphore(0)`` would deadlock).
    """
    if cli_jobs is not None:
        return max(1, cli_jobs)
    raw = profile_cfg.get("jobs", 1) if hasattr(profile_cfg, "get") else 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _watch_path_for(workspace_arg: Path | None, submitted: list[Any]) -> str:
    """Pick the argument to suggest for ``molexp explore`` reopen messages.

    Prefers an explicit ``--workspace`` value; otherwise derives the
    workspace root from the first submitted run. The returned string
    is cwd-relative when possible (so it matches the ``workspace``
    argument the user would type), else absolute.
    """
    root: Path | None = None
    if workspace_arg is not None:
        root = Path(workspace_arg)
    elif submitted:
        try:
            root = submitted[0].experiment.project.workspace.root
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


def _execute_sweep(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    resume: bool,
    workspace: Path | None,
    run_handler: RunHandler | None,
    mode_label: str,
    suppress_ok: bool = False,
    jobs: int = 1,
    use_sweep_graph: bool = False,
) -> tuple[int, list[Any]]:
    """Walk each workspace's experiments and dispatch one run per replica.

    Idempotent: re-running with the same script produces deterministic
    run IDs (derived from parameters + active profile) so repeated
    invocations skip already-executed replicas unless ``--resume`` is
    set, in which case non-succeeded runs of the matching profile are
    replayed.
    """
    from molexp.entry import load_workspaces
    from molexp.workspace.workspace import set_cli_root_override

    override_path = Path(workspace).resolve() if workspace is not None else None
    set_cli_root_override(override_path)
    try:
        workspaces = load_workspaces(script)
    except Exception as exc:
        rprint(f"[red]Error importing {script.name}:[/red] {exc}")
        rprint(traceback.format_exc(), end="")
        raise typer.Exit(1)
    finally:
        set_cli_root_override(None)

    if not workspaces:
        rprint(
            "[red]Error:[/red] No me.entry() call found in script. "
            "Add [bold]me.entry(workspace)[/bold] at module level."
        )
        raise typer.Exit(1)

    total_dispatched = 0
    dispatched_runs: list[Any] = []
    all_replicas: list[tuple[Any, Any, Any]] = []

    for ws in workspaces:
        if override_path is not None and ws.root == override_path:
            rprint(f"[dim]--workspace override active: {ws.root}[/dim]")

        for project in ws.registered_projects():
            for exp in project.registered_experiments():
                if exp.workflow is None:
                    rprint(
                        f"[red]Error:[/red] Experiment {exp.name!r} has no workflow. "
                        "Call experiment.set_workflow(fn) before me.entry()."
                    )
                    raise typer.Exit(1)

                seeds = exp.get_seeds()
                total = exp.n_replicas
                label = f"[cyan]resume[/cyan] + {mode_label}" if resume else mode_label
                profile_display = profile_cfg.name or "(defaults)"
                rprint(
                    f"\n[bold]Experiment:[/bold] {exp.name}"
                    f"\n  Script:    {script}"
                    f"\n  Workspace: {ws.root}"
                    f"\n  Project:   {project.name}"
                    f"\n  Profile:   {profile_display}"
                    f"\n  Runs:      {total} replicas"
                    f"\n  Mode:      {label}"
                )

                created_runs: list[tuple[Any, int]] = []

                for replica_idx, seed in enumerate(seeds):
                    run_params = {**exp.params, "seed": seed, "replica": replica_idx}
                    # Profile name and config hash are folded into the ID
                    # so different profiles of the same replica do not
                    # collide under the same run directory.
                    id_seed = dict(run_params)
                    if profile_cfg.name is not None:
                        id_seed["_profile"] = profile_cfg.name
                        id_seed["_config_hash"] = profile_cfg.content_hash()
                    run_id = deterministic_run_id(id_seed)

                    existing = exp.get_run(run_id)
                    if existing is not None and existing.status == "running":
                        if reap_zombie_run(existing):
                            rprint(
                                f"  [yellow]![/yellow] {exp.id}  seed={seed} "
                                "(stale 'running' run reaped → failed)"
                            )

                    if resume:
                        if existing is None:
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (no existing run, skipped)[/dim]"
                            )
                            continue
                        # Resume all non-succeeded runs that match this profile.
                        if existing.status == "succeeded":
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (already succeeded, skipped)[/dim]"
                            )
                            continue
                        if existing.metadata.profile != profile_cfg.name:
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (profile mismatch: "
                                f"{existing.metadata.profile!r} vs {profile_cfg.name!r}, skipped)[/dim]"
                            )
                            continue
                        mol_run = existing
                    elif existing is not None:
                        if existing.status in ("succeeded", "running"):
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} ({existing.status}, skipped)[/dim]"
                            )
                            continue
                        mol_run = existing
                    else:
                        mol_run = exp.run(parameters=run_params, id=run_id)

                    created_runs.append((mol_run, seed))
                    icon = "[cyan]>[/cyan]" if resume else "[dim]o[/dim]"
                    rprint(f"  {icon} {exp.id}  seed={seed}")

                submit_cwd_str = str(Path.cwd().resolve())
                for mol_run, _seed in created_runs:
                    # Persist script path, submit cwd, and profile config into
                    # run.json before any handler so the worker can reconstruct
                    # everything from run_dir alone (via ``molexp execute
                    # <run_dir>``).  ``submit_cwd`` lets the worker resolve
                    # cwd-relative paths in the user script the same way they
                    # resolved at submit time.
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

                total_dispatched += len(created_runs)

    if use_sweep_graph:
        # ``--local`` and the cluster backends share the same dispatch model:
        # each replica is a ``molexp execute`` subprocess submitted through
        # molq with ``cwd=exec_dir``.  This gives ``--local`` per-attempt
        # isolation (cwd-relative output like ``logs/...`` lands inside
        # ``executions/<exec_id>/``) and a single job-state model across
        # backends so ``molexp watch`` / ``molexp explore`` work uniformly.
        from molexp.plugins.submit_molq.local_sweep import run_local_sweep
        from molexp.sweep import SweepReplica

        replicas = [
            SweepReplica(mol_run=mol_run, experiment=exp) for mol_run, exp, _project in all_replicas
        ]
        if replicas:
            failures = asyncio.run(run_local_sweep(replicas, jobs=jobs))
            for rid, exc in failures.items():
                rprint(f"[red]Run {rid} failed:[/red] {exc}")
        dispatched_runs.extend(mol_run for mol_run, _, _ in all_replicas)
        if not suppress_ok:
            verb = "resumed" if resume else "completed"
            rprint(f"\n[green]OK[/green] {total_dispatched} runs {verb}.")
    else:
        if run_handler is None:
            raise RuntimeError("run_handler must be provided when use_sweep_graph=False")
        for mol_run, exp, project in all_replicas:
            run_handler(script, mol_run, exp, project)
            dispatched_runs.append(mol_run)
            rprint(f"  [cyan]>[/cyan] dispatched {exp.id}  run={mol_run.id}")
        if not suppress_ok:
            verb = "resumed" if resume else "completed"
            rprint(f"\n[green]OK[/green] {total_dispatched} runs {verb}.")

    return total_dispatched, dispatched_runs


# ── Background launcher ─────────────────────────────────────────────────────


def _launch_bg(
    *,
    script: Path,
    config_path: Path | None,
    profile: str | None,
    overrides: list[str],
    resume: bool,
    workspace: Path | None,
) -> None:
    """Fork a detached subprocess that re-runs the sweep without ``--bg``."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "molexp.cli", "run", str(script), "--local"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if profile is not None:
        cmd.extend(["--profile", profile])
    for override in overrides:
        cmd.extend(["--override", override])
    if resume:
        cmd.append("--resume")
    if workspace is not None:
        cmd.extend(["--workspace", str(workspace)])

    ws_root = Path(workspace).resolve() if workspace else Path.cwd()
    ws_root.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=open(ws_root / "molexp_bg.log", "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    rprint(
        f"[green]OK[/green] Launched in background (pid={proc.pid}). "
        f"Logs: [bold]{ws_root / 'molexp_bg.log'}[/bold]"
    )


# ── Typer argument / option aliases ─────────────────────────────────────────

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


# ── Command ─────────────────────────────────────────────────────────────────


@app.command()
def run(
    script: _SCRIPT_ARG,
    # ── Execution backend ──────────────────────────────────────────────────
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Run locally, sequentially (default).",
            rich_help_panel="Execution Backend",
        ),
    ] = False,
    scheduler: Annotated[
        Optional[str],
        typer.Option(
            "--scheduler",
            help="Submit via a molq scheduler backend (e.g. local, slurm, pbs, lsf).",
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    # ── Config / profile ───────────────────────────────────────────────────
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help=(
                "Path to molcfg file (YAML or JSON). "
                "Defaults to ./molcfg.yaml / .yml / .json if present."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    profile: Annotated[
        Optional[str],
        typer.Option(
            "--profile",
            help=(
                "Named molcfg profile to activate (e.g. 'dry-run', 'smoke'). "
                "Profile names with '-' are normalized to '_'."
            ),
        ),
    ] = None,
    overrides: Annotated[
        Optional[list[str]],
        typer.Option(
            "--override",
            help=(
                "Override a config key after profile resolution. "
                "Format: KEY=VALUE (repeatable). "
                "Dot notation supported for nested keys: model.n_layers=3. "
                "Values are auto-coerced to bool/int/float/str."
            ),
        ),
    ] = None,
    # ── Common options ─────────────────────────────────────────────────────
    jobs: Annotated[
        Optional[int],
        typer.Option(
            "--jobs",
            "-j",
            help=(
                "Maximum number of replicas to run concurrently (local backend "
                "only). If omitted, falls back to the active profile's "
                "``jobs:`` field, or ``1`` (serial) when unset."
            ),
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Re-execute non-succeeded runs whose profile matches the one selected by --profile."
            ),
        ),
    ] = False,
    bg: Annotated[
        bool,
        typer.Option(
            "--bg", help="Run in background (local only). Logs to <workspace>/molexp_bg.log."
        ),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── HPC options (slurm / pbs / lsf) ───────────────────────────────────
    cpus: Annotated[
        Optional[int],
        typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="HPC Options"),
    ] = None,
    mem: Annotated[
        Optional[str],
        typer.Option(
            "--mem", help="Memory per job (e.g. 8G, 512M).", rich_help_panel="HPC Options"
        ),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option(
            "--time",
            "-t",
            help="Wall-clock time limit (e.g. 12:00:00, 2h30m).",
            rich_help_panel="HPC Options",
        ),
    ] = None,
    gpus: Annotated[
        Optional[int],
        typer.Option("--gpus", help="GPUs per job.", rich_help_panel="HPC Options"),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option(
            "--gpu-type", help="GPU type constraint (e.g. a100).", rich_help_panel="HPC Options"
        ),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option(
            "--account", "-A", help="Account / project name.", rich_help_panel="HPC Options"
        ),
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name.", rich_help_panel="HPC Options"),
    ] = None,
    target: Annotated[
        Optional[str],
        typer.Option(
            "--target",
            help=(
                "Named compute target from `molexp target list`. Overrides "
                "--scheduler/--cluster — the target carries both axes "
                "(transport: local/ssh + scheduler: shell/slurm/pbs/lsf)."
            ),
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    # ── SLURM-specific ─────────────────────────────────────────────────────
    partition: Annotated[
        Optional[str],
        typer.Option("--partition", "-p", help="SLURM partition.", rich_help_panel="SLURM Options"),
    ] = None,
    qos: Annotated[
        Optional[str],
        typer.Option("--qos", help="SLURM QOS.", rich_help_panel="SLURM Options"),
    ] = None,
    # ── PBS / LSF-specific ─────────────────────────────────────────────────
    queue: Annotated[
        Optional[str],
        typer.Option(
            "--queue", "-q", help="PBS/LSF queue name.", rich_help_panel="PBS / LSF Options"
        ),
    ] = None,
    # ── Monitor ────────────────────────────────────────────────────────────
    block: Annotated[
        bool,
        typer.Option(
            "--block",
            help=(
                "Block after cluster submission and open the interactive "
                "run monitor until all jobs finish (press q to close)."
            ),
            rich_help_panel="HPC Options",
        ),
    ] = False,
) -> None:
    """Execute the workflow(s) defined by *script*."""
    # ── Backend selection ──────────────────────────────────────────────────
    backend_flags = sum(
        1 for f in (local, scheduler is not None, target is not None) if f
    )
    if backend_flags > 1:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag "
            "(--local, --scheduler, --target)."
        )
        raise typer.Exit(1)

    # --target resolves the scheduler from the target's metadata.
    selected_target = None
    if target is not None:
        from molexp.workspace import Workspace, get_target

        ws_path = workspace if workspace is not None else Path.cwd()
        ws = Workspace(ws_path)
        try:
            selected_target = get_target(ws, target)
        except KeyError as exc:
            rprint(f"[red]{exc}[/red] — see `molexp target list`.")
            raise typer.Exit(1) from exc

    selected_scheduler = (
        selected_target.scheduler if selected_target is not None else scheduler
    )
    is_local = selected_scheduler is None and selected_target is None

    if bg and not is_local:
        rprint("[red]Error:[/red] --bg is only valid with --local.")
        raise typer.Exit(1)

    profile_cfg = _resolve_profile(config, profile)
    profile_cfg = _apply_overrides(profile_cfg, overrides or [])

    # ── Local execution ────────────────────────────────────────────────────
    if is_local:
        if bg:
            _launch_bg(
                script=script,
                config_path=config,
                profile=profile,
                overrides=overrides or [],
                resume=resume,
                workspace=workspace,
            )
            return

        profile_label = (
            f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
        )
        label = f"[green]local[/green] profile={profile_label}"

        effective_jobs = _resolve_jobs(jobs, profile_cfg)
        _execute_sweep(
            script=script,
            profile_cfg=profile_cfg,
            resume=resume,
            workspace=workspace,
            run_handler=None,
            mode_label=label,
            jobs=effective_jobs,
            use_sweep_graph=True,
        )
        return

    # ── Cluster backends ───────────────────────────────────────────────────
    # The local branch above returned, so a scheduler must have been selected
    # to reach this point. Narrow `selected_scheduler` for the type checker.
    assert selected_scheduler is not None
    from molexp.plugins.submit_molq.metadata import supported_schedulers
    from molexp.plugins.submit_molq.submit import make_submit_handler

    available_schedulers = supported_schedulers()
    if available_schedulers and selected_scheduler not in available_schedulers:
        supported_text = ", ".join(available_schedulers)
        rprint(
            f"[red]Error:[/red] Unsupported molq scheduler: {selected_scheduler!r}. "
            f"Available: {supported_text}"
        )
        raise typer.Exit(1)

    selected_queue = partition if partition is not None else queue
    handler = make_submit_handler(
        scheduler=selected_scheduler,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "gpu_type": gpu_type, "time": time},
        scheduling={"queue": selected_queue, "account": account, "qos": qos},
        target=selected_target,
    )
    profile_label = (
        f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
    )
    mode_label = f"[magenta]{selected_scheduler}[/magenta] profile={profile_label}"

    n, submitted = _execute_sweep(
        script=script,
        profile_cfg=profile_cfg,
        resume=resume,
        workspace=workspace,
        run_handler=handler,
        mode_label=mode_label,
        suppress_ok=True,
    )

    if n == 0:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[dim]No runs {verb}.[/dim]")
        return

    watch_arg = _watch_path_for(workspace, submitted)

    if block and submitted:
        from molexp.monitor import RunMonitor

        rprint(f"\n[dim]Submitted {n} runs. Opening monitor… (press q to close)[/dim]")
        RunMonitor(title=f"{script.stem}  [{mode_label}]").watch(submitted)
        rprint(f"\n[dim]Monitor closed. {n} runs are still executing (if any).[/dim]")
        rprint(f"[dim]Reopen with:  molexp explore {watch_arg}[/dim]")
    else:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[green]OK[/green] {n} runs {verb}.")
        if submitted:
            rprint(f"[dim]Open the explorer with:  molexp explore {watch_arg}[/dim]")


@app.command()
def execute(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the run directory (contains run.json).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
            help="Pre-allocated execution_id (used by molq submission to "
            "guarantee stdout/stderr/jobs land in the same directory the "
            "worker writes workflow.json into).",
        ),
    ] = None,
) -> None:
    """Execute a run from its run directory.

    Worker entry point used by cluster backends.  The run's
    ``workflow_snapshot.entrypoint`` (``"<file>:<qualname>"``) is the
    sole source of truth — the file is imported as a *non*-``__main__``
    module so any ``if __name__ == "__main__":`` guard in the user
    script skips.  Before the import the worker chdirs to
    ``run.metadata.submit_cwd`` so any cwd-relative paths in
    module-level code (e.g. ``Workspace("./lab")``) resolve to the same
    location they did at submit time; cwd is restored afterwards so the
    job continues to run with ``cwd=exec_dir`` (the per-attempt
    directory the scheduler launched the worker in).  This means
    cwd-relative file output from user tasks (e.g. ``logs/train.log``,
    ``logs/tensorboard/``) lands inside ``executions/<exec_id>/`` —
    co-located with stdout/stderr/workflow.json — instead of leaking
    into the run directory and being overwritten on retry.
    """
    import asyncio
    import os

    from molexp.config import ProfileConfig
    from molexp.entry import load_workflow_from_entrypoint
    from molexp.workflow.spec import WorkflowSpec
    from molexp.workspace.experiment import _promote_to_workflow
    from molexp.workspace.run import RunContext

    ctx = RunContext.open(run_dir)
    run = ctx.run

    snapshot = run.metadata.workflow_snapshot
    if snapshot is None or not snapshot.entrypoint:
        rprint(
            "[red]Error:[/red] run.json has no 'workflow_snapshot.entrypoint'. "
            "Re-submit the run with a current version of molexp so the "
            "worker can locate the workflow."
        )
        raise typer.Exit(1)

    saved_cwd = os.getcwd()
    submit_cwd = run.metadata.submit_cwd
    chdir_target: Path | None = None
    if submit_cwd:
        candidate = Path(submit_cwd)
        if candidate.is_dir():
            chdir_target = candidate
        else:
            rprint(
                f"[yellow]Warning:[/yellow] submit_cwd {submit_cwd!r} not "
                "reachable on this worker; importing with the current cwd. "
                "Module-level relative paths in the workflow file may "
                "resolve unexpectedly."
            )
    try:
        if chdir_target is not None:
            os.chdir(chdir_target)
        try:
            target = load_workflow_from_entrypoint(snapshot.entrypoint)
        except Exception as exc:
            rprint(f"[red]Error loading {snapshot.entrypoint}:[/red] {exc}")
            rprint(traceback.format_exc(), end="")
            raise typer.Exit(1)
    finally:
        os.chdir(saved_cwd)

    if isinstance(target, WorkflowSpec):
        workflow = target
    elif callable(target):
        workflow = _promote_to_workflow(target, run.experiment.name)
    else:
        rprint(
            f"[red]Error:[/red] entrypoint {snapshot.entrypoint!r} resolved to "
            f"{type(target).__name__}, expected WorkflowSpec or callable."
        )
        raise typer.Exit(1)

    profile_config = ProfileConfig(run.metadata.config, name=run.metadata.profile)
    asyncio.run(
        workflow.execute(
            run=run,
            profile_config=profile_config,
            execution_id=execution_id,
        )
    )


# Silence unused-import lint (console is re-exported for other modules).
_ = console
