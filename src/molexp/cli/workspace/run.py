"""``molexp {run,exec,shell}`` — execution commands."""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from molexp._typing import JSONValue
from molexp.cli._app import app
from molexp.cli._common import deterministic_run_id, reap_zombie_run, rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.profile import MolCfg, ProfileConfig, load_molcfg
from molexp.profile.loader import find_default_config
from molexp.workflow import Workflow
from molexp.workspace.target import LocalTarget, RemoteTarget

if TYPE_CHECKING:
    from molexp.workflow.protocols import RunContextLike
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run

RunHandler = Callable[["Path", "Run", "Experiment", "Project"], None]

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
    from molexp.entry import load_workspaces
    from molexp.workspace.workspace import set_cli_root_override

    override_path = Path(workspace).resolve() if workspace is not None else None
    set_cli_root_override(override_path)
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

    total_dispatched = 0
    dispatched_runs: list[Run] = []
    all_replicas: list[tuple[Run, Experiment, Project]] = []

    for ws in workspaces:
        if override_path is not None and ws.root == override_path:
            rprint(f"[dim]--workspace override active: {ws.root}[/dim]")
        for project in ws.list_projects():
            for exp in project.list_experiments():
                if Workflow.for_experiment(exp) is None:
                    continue
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
                    id_seed = dict(run_params)
                    if profile_cfg.name is not None:
                        id_seed["_profile"] = profile_cfg.name
                        id_seed["_config_hash"] = profile_cfg.content_hash()
                    run_id = deterministic_run_id(id_seed)
                    from molexp.workspace import RunNotFoundError as _RunNotFound

                    try:
                        existing = exp.get_run(run_id)
                    except _RunNotFound:
                        existing = None
                    if existing is not None and existing.status == "running":  # noqa: SIM102
                        if reap_zombie_run(existing):
                            rprint(
                                f"  [yellow]![/yellow] {exp.id}  seed={seed} (stale 'running' run reaped → failed)"
                            )
                    if resume:
                        if existing is None:
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (no existing run, skipped)[/dim]"
                            )
                            continue
                        if existing.status == "succeeded":
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (already succeeded, skipped)[/dim]"
                            )
                            continue
                        if existing.metadata.profile != profile_cfg.name:
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (profile mismatch, skipped)[/dim]"
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
                        mol_run = exp.add_run(parameters=run_params, id=run_id)
                    created_runs.append((mol_run, seed))
                    icon = "[cyan]>[/cyan]" if resume else "[dim]o[/dim]"
                    rprint(f"  {icon} {exp.id}  seed={seed}")

                submit_cwd_str = str(Path.cwd().resolve())
                for mol_run, _seed in created_runs:
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


def _make_local_inprocess_handler(profile_cfg: ProfileConfig) -> RunHandler:
    import asyncio

    from molexp.workspace.run import RunContext

    def _handler(_script: Path, mol_run: Run, experiment: Experiment, _project: Project) -> None:
        spec = Workflow.for_experiment(experiment)
        if spec is None:
            raise RuntimeError(f"Experiment {experiment.name!r} has no workflow attached.")
        with RunContext(mol_run, profile_config=profile_cfg) as ctx:
            asyncio.run(spec.execute(run_context=cast("RunContextLike", ctx)))

    return _handler


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
        bool, typer.Option("--resume", help="Re-execute non-succeeded runs.")
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
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        rprint("[red]Error:[/red] 'run' on remote targets is not yet supported.")
        rprint("  Use [bold]molexp exec[/bold] or [bold]shell[/bold] for remote execution.")
        raise typer.Exit(1)

    backend_flags = sum(1 for f in (local, scheduler is not None, target_cli is not None) if f)
    if backend_flags > 1:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag (--local, --scheduler, --target)."
        )
        raise typer.Exit(1)

    selected_target = None
    if target_cli is not None:
        from molexp.workspace import Workspace, get_target

        ws = Workspace(target.path)
        try:
            selected_target = get_target(ws, target_cli)
        except KeyError as exc:
            rprint(f"[red]{exc}[/red] — see `molexp target list`.")
            raise typer.Exit(1) from exc

    selected_scheduler = selected_target.scheduler if selected_target is not None else scheduler
    is_local = selected_scheduler is None and selected_target is None

    profile_cfg = _resolve_profile(config, profile)
    profile_cfg = _apply_overrides(profile_cfg, overrides or [])

    if is_local:
        profile_label = (
            f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
        )
        mode_label = f"[green]local[/green] profile={profile_label}"
        _dispatch_runs(
            script=script,
            profile_cfg=profile_cfg,
            resume=resume,
            workspace=target.path,
            run_handler=_make_local_inprocess_handler(profile_cfg),
            mode_label=mode_label,
        )
        return

    from molexp.plugins.submit_molq.metadata import supported_schedulers
    from molexp.plugins.submit_molq.submit import make_submit_handler

    available_schedulers = supported_schedulers()
    if available_schedulers and selected_scheduler not in available_schedulers:
        supported_text = ", ".join(available_schedulers)
        rprint(
            f"[red]Error:[/red] Unsupported molq scheduler: {selected_scheduler!r}. Available: {supported_text}"
        )
        raise typer.Exit(1)

    selected_queue = partition if partition is not None else queue
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
        workspace=target.path,
        run_handler=handler,
        mode_label=mode_label,
        suppress_ok=True,
    )
    if n == 0:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[dim]No runs {verb}.[/dim]")
        return

    watch_arg = _watch_path_for(target.path, submitted)
    if block and submitted:
        from molexp.monitor import RunMonitor

        rprint(f"\n[dim]Submitted {n} runs. Opening monitor… (press q to close)[/dim]")
        RunMonitor(title=f"{script.stem}  [{mode_label}]").watch(submitted)
        rprint(f"\n[dim]Monitor closed. {n} runs are still executing (if any).[/dim]")
        rprint(f"[dim]Reopen with:  molexp monitor -t {watch_arg}[/dim]")
    else:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[green]OK[/green] {n} runs {verb}.")
        if submitted:
            rprint(f"[dim]Monitor runs with:  molexp monitor -t {watch_arg}[/dim]")


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
