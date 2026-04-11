"""Script-based experiment runner helpers.

Provides ``ExperimentDef`` (declarative sweep spec) and ``standalone_run``
(local execution entry point for direct script invocation).

Typical usage in an experiment script::

    from molexp.runner import ExperimentDef, standalone_run
    from molexp.workspace import GridSpace, RunContext

    EXPERIMENT = ExperimentDef(
        name="my-sweep",
        project="my-project",
        param_space=GridSpace({"lr": [1e-4, 3e-4], "l_max": [1, 2]}),
        n_replicas=3,
        workspace_root="./workspace",
    )

    def train(ctx: RunContext) -> None:
        params = ctx.run.parameters
        ...

    if __name__ == "__main__":
        standalone_run(EXPERIMENT, train)

When invoked as ``molexp run script.py``, the CLI discovers ``EXPERIMENT``
and ``train`` (or any callable registered via ``EXPERIMENT.entry_point``) and
orchestrates workspace creation, run registration, and execution/submission.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from molexp.workspace import GridSpace, ParamSpace, Workspace
from molexp.workspace.run import RunContext

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


@dataclass
class ExperimentDef:
    """Declarative experiment definition embedded in a training script.

    Attributes:
        name: Unique sweep name (also used as the project name default).
        project: Project name within the workspace.
        param_space: Parameter space to sweep (``GridSpace``, ``UniformSpace``, etc.).
        n_replicas: Number of independent repeats per parameter combination.
        workspace_root: Root path for the molexp workspace.
        description: Human-readable description shown in the UI.
        tags: Metadata tags attached to runs.
        seeds: Explicit seeds for each replica.  Defaults to ``[42, 123, 456, …]``.
        entry_point: Name of the callable in the script to invoke per run.
            Defaults to ``"train"`` — must accept ``(ctx: RunContext) -> None``.
    """

    name: str
    project: str
    param_space: ParamSpace
    n_replicas: int = 3
    workspace_root: Path = field(default_factory=lambda: Path("./workspace"))
    description: str = ""
    tags: list[str] = field(default_factory=list)
    seeds: list[int] | None = None
    entry_point: str = "train"

    def get_seeds(self) -> list[int]:
        """Return the replica seeds (length == n_replicas)."""
        if self.seeds is not None:
            return list(self.seeds[: self.n_replicas])
        seeds = list(_DEFAULT_SEEDS)
        while len(seeds) < self.n_replicas:
            seeds.append(seeds[-1] + 111)
        return seeds[: self.n_replicas]


def standalone_run(
    experiment_def: ExperimentDef,
    train_fn: Callable[[RunContext], None],
    workspace_root: Path | str | None = None,
) -> None:
    """Run the full parameter sweep locally (direct-script mode).

    Called from the ``if __name__ == "__main__"`` block when the script is
    invoked without ``--run-dir``.  Creates the workspace/project/experiments,
    registers all runs, then executes each one sequentially.

    Args:
        experiment_def: The sweep specification.
        train_fn: Training function accepting a :class:`RunContext`.
        workspace_root: Override for ``experiment_def.workspace_root``.
    """
    from rich import print as rprint

    root = Path(workspace_root or experiment_def.workspace_root).resolve()
    workspace = Workspace.from_path(root)

    project = workspace.get_project(experiment_def.project)
    if project is None:
        project = workspace.create_project(name=experiment_def.project)

    seeds = experiment_def.get_seeds()
    total = len(experiment_def.param_space) * experiment_def.n_replicas
    done = 0

    for params in experiment_def.param_space:
        exp_id = _params_to_id(params)
        experiment = project.get_experiment(exp_id)
        if experiment is None:
            experiment = project.create_experiment(
                name=_params_to_label(params),
                id=exp_id,
                workflow_source=experiment_def.name,
                parameter_space=dict(params),
            )

        for replica_idx, seed in enumerate(seeds):
            run_params = {**params, "seed": seed, "replica": replica_idx}
            run = experiment.create_run(parameters=run_params)
            done += 1
            rprint(
                f"[cyan]▶[/cyan] [{done}/{total}] "
                f"{exp_id} seed={seed}"
            )
            with run.start() as ctx:
                train_fn(ctx)


def _params_to_id(params: dict[str, Any]) -> str:
    """Compact, filesystem-safe experiment ID from a parameter dict."""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            # 1e-4 → 1e-04, 3e-4 → 3e-04
            formatted = f"{v:.0e}".replace("+", "")
            parts.append(f"{k}-{formatted}")
        else:
            parts.append(f"{k}-{v}")
    return "_".join(parts)


def _params_to_label(params: dict[str, Any]) -> str:
    """Human-readable experiment label."""
    parts = []
    for k, v in sorted(params.items()):
        parts.append(f"{k}={v}")
    return ", ".join(parts)
