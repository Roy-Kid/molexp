"""Quick start — workspace + experiment + tracked run, end to end.

Matches ``docs/getting-started/quick-start.md``.

Run directly::

    python examples/getting_started/01_quick_start.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="train")


@wf.task
async def train(lr: float = 1e-3, epochs: int = 3) -> dict:
    """Root task — its inputs arrive bound to named parameters.

    The engine fills ``lr`` from the run's sweep params (``params={"lr": [1e-3]}``)
    and ``epochs`` from build-time config; each falls back to its declared
    default when absent. ``ctx`` is omitted because this body needs no
    ``ctx.workdir`` scratch space — the only data surface left on the context.
    """
    final_loss = 1.0 / (epochs * (lr * 1000 + 1))
    return {"lr": lr, "epochs": epochs, "final_loss": final_loss}


compiled = wf.compile()


async def main() -> None:
    workspace_root = Path(tempfile.mkdtemp(prefix="molexp-quickstart-"))
    print(f"workspace root: {workspace_root}")

    # Declare: one experiment running `compiled` over a one-cell sweep.
    ws = me.Workspace(workspace_root, name="quickstart")
    experiment = ws.project("demo").experiment("train").run(compiled, params={"lr": [1e-3]})

    # Drive the seeded run in-process (`molexp run` does this for you).
    run = experiment.list_runs()[0]
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
        ctx.set_result("final_loss", result.outputs["train"]["final_loss"])

    print(f"status:     {result.status}")
    print(f"final_loss: {run.get_result('final_loss')}")
    print(f"run_dir:    {run.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
