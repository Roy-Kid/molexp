"""``molexp run`` — execute workflows defined by a Python script."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from molexp._typing import JSONValue
from molexp.config import MolCfg, ProfileConfig, load_molcfg
from molexp.config.loader import find_default_config

from . import app
from ._common import (
    console,
    deterministic_run_id,
    reap_zombie_run,
    rprint,
)

if TYPE_CHECKING:
    from molexp.workflow.protocols import RunContextLike
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run

RunHandler = Callable[["Path", "Run", "Experiment", "Project"], None]


# ── Config / profile resolution ─────────────────────────────────────────────


def _coerce_value(raw: str) -> JSONValue:
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


def _set_nested(d: dict[str, JSONValue], key_path: str, value: JSONValue) -> None:
    """Set a nested dict value using dot-notation key path (mutates *d*).

    Walks the dot-separated path, creating intermediate ``dict`` cells when
    a level is absent or holds a non-dict value (a leaf scalar / list).
    The narrowing dance via ``isinstance`` keeps ty happy under
    ``JSONValue``'s recursive shape.
    """
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


# ── Replica dispatch ────────────────────────────────────────────────────────


def _watch_path_for(workspace_arg: Path | None, submitted: list[Run]) -> str:
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


def _dispatch_runs(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    resume: bool,
    workspace: Path | None,
    run_handler: RunHandler,
    mode_label: str,
    suppress_ok: bool = False,
) -> tuple[int, list[Run]]:
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
    dispatched_runs: list[Run] = []
    all_replicas: list[tuple[Run, Experiment, Project]] = []

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

                created_runs: list[tuple[Run, int]] = []

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

    for mol_run, exp, project in all_replicas:
        run_handler(script, mol_run, exp, project)
        dispatched_runs.append(mol_run)
        rprint(f"  [cyan]>[/cyan] dispatched {exp.id}  run={mol_run.id}")
    if not suppress_ok:
        verb = "resumed" if resume else "completed"
        rprint(f"\n[green]OK[/green] {total_dispatched} runs {verb}.")

    return total_dispatched, dispatched_runs


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


def _make_local_inprocess_handler(profile_cfg: ProfileConfig) -> RunHandler:
    """Build an in-process replica handler — no scheduler, no subprocess.

    Used by ``molexp run --local`` so the workflow runs in the parent
    Python process. Each replica opens its own :class:`RunContext` (with
    the active profile config so ``run.metadata.profile`` is preserved)
    and awaits :meth:`WorkflowSpec.execute`; failures bubble up as
    exceptions and are caught by the dispatcher.
    """
    import asyncio

    from molexp.workspace.run import RunContext

    def _handler(_script: Path, mol_run: Run, experiment: Experiment, _project: Project) -> None:
        spec = experiment.workflow
        if spec is None:
            raise RuntimeError(f"Experiment {experiment.name!r} has no workflow attached.")
        with RunContext(mol_run, profile_config=profile_cfg) as ctx:
            # ``RunContext`` structurally satisfies ``RunContextLike``;
            # ty's protocol-attribute matching is conservative, so the
            # cross-layer cast acknowledges the duck-typed boundary.
            asyncio.run(spec.execute(run_context=cast("RunContextLike", ctx)))

    return _handler


@app.command()
def run(
    script: _SCRIPT_ARG,
    # ── Execution backend ──────────────────────────────────────────────────
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Run replicas in-process, sequentially (no scheduler).",
            rich_help_panel="Execution Backend",
        ),
    ] = False,
    scheduler: Annotated[
        str | None,
        typer.Option(
            "--scheduler",
            help="Submit via a molq scheduler backend (e.g. local, slurm, pbs, lsf).",
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    # ── Config / profile ───────────────────────────────────────────────────
    config: Annotated[
        Path | None,
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
        str | None,
        typer.Option(
            "--profile",
            help=(
                "Named molcfg profile to activate (e.g. 'dry-run', 'smoke'). "
                "Profile names with '-' are normalized to '_'."
            ),
        ),
    ] = None,
    overrides: Annotated[
        list[str] | None,
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
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Re-execute non-succeeded runs whose profile matches the one selected by --profile."
            ),
        ),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── HPC options (slurm / pbs / lsf) ───────────────────────────────────
    cpus: Annotated[
        int | None,
        typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="HPC Options"),
    ] = None,
    mem: Annotated[
        str | None,
        typer.Option(
            "--mem", help="Memory per job (e.g. 8G, 512M).", rich_help_panel="HPC Options"
        ),
    ] = None,
    time: Annotated[
        str | None,
        typer.Option(
            "--time",
            "-t",
            help="Wall-clock time limit (e.g. 12:00:00, 2h30m).",
            rich_help_panel="HPC Options",
        ),
    ] = None,
    gpus: Annotated[
        int | None,
        typer.Option("--gpus", help="GPUs per job.", rich_help_panel="HPC Options"),
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
    target: Annotated[
        str | None,
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
        str | None,
        typer.Option("--partition", "-p", help="SLURM partition.", rich_help_panel="SLURM Options"),
    ] = None,
    qos: Annotated[
        str | None,
        typer.Option("--qos", help="SLURM QOS.", rich_help_panel="SLURM Options"),
    ] = None,
    # ── PBS / LSF-specific ─────────────────────────────────────────────────
    queue: Annotated[
        str | None,
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
    backend_flags = sum(1 for f in (local, scheduler is not None, target is not None) if f)
    if backend_flags > 1:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag (--local, --scheduler, --target)."
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

    selected_scheduler = selected_target.scheduler if selected_target is not None else scheduler
    is_local = selected_scheduler is None and selected_target is None

    profile_cfg = _resolve_profile(config, profile)
    profile_cfg = _apply_overrides(profile_cfg, overrides or [])

    # ── In-process execution (default / --local) ──────────────────────────
    if is_local:
        profile_label = (
            f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
        )
        mode_label = f"[green]local[/green] profile={profile_label}"
        _dispatch_runs(
            script=script,
            profile_cfg=profile_cfg,
            resume=resume,
            workspace=workspace,
            run_handler=_make_local_inprocess_handler(profile_cfg),
            mode_label=mode_label,
        )
        return

    # ── Cluster backends ───────────────────────────────────────────────────
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
    # Past the ``is_local`` early-return above, ``selected_scheduler`` is
    # guaranteed non-None — narrow ``str | None → str`` for the call.
    assert selected_scheduler is not None
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

    n, submitted = _dispatch_runs(
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

    with ctx:
        asyncio.run(
            workflow.execute(
                run_context=cast("RunContextLike", ctx),
                execution_id=execution_id,
            )
        )


# Silence unused-import lint (console is re-exported for other modules).
_ = console
