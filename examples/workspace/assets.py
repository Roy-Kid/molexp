"""Artifacts, logs, checkpoints, data imports, and catalog queries.

Matches ``docs/guide/assets.md``.

Walks through:

1. ``ctx.artifact.save`` — writes a file and registers an ``ArtifactAsset``.
2. ``ctx.log(name).append`` — appends to a ``LogAsset`` scoped to the run.
3. ``ctx.checkpoint`` — writes a ``CheckpointAsset`` with parent chaining.
4. ``ws.data_assets.import_asset`` — pulls external data into the workspace.
5. ``ctx.find_asset`` — scope-walking lookup (run → experiment → project →
   workspace) on the driver-side ``RunContext``.
6. ``ws.catalog.query_assets`` — workspace-wide asset queries.

Task bodies stay on the pure ``{inputs, config}`` contract; the asset
helpers live on the ``RunContext`` the driver opened via ``run.start()``.

Run directly::

    python examples/workspace/assets.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="train")


@wf.task
async def train(ctx: TaskContext) -> dict:
    return {"loss": 0.08, "acc": 0.94}


compiled = wf.compile()


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-assets-"))
    ws = me.Workspace(root, name="assets-demo")

    # 4. Import a workspace-scoped dataset from outside.
    external = root / "external.csv"
    external.write_text("x,y\n1,2\n3,4\n")
    ws.data_assets.import_asset("toy-dataset", external)

    exp = ws.project("demo").experiment("train").run(compiled, params=None)
    run = exp.list_runs()[0]
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)

        # 1. Artifact — arbitrary payload written to run_dir/artifacts/.
        ctx.artifact.save("metrics.json", result.outputs["train"])

        # 2. Log — line-oriented, appendable, scoped to this run.
        log = ctx.log("train")
        log.append("epoch 1 start")
        log.append("epoch 1 done  loss=0.10")
        log.append("epoch 2 done  loss=0.08")

        # 3. Checkpoints — chained via parent_ckpt_id.
        ctx.checkpoint("epoch-1", data={"step": 1})
        ctx.checkpoint("epoch-2", data={"step": 2})

        # 5. find_asset walks run → experiment → project → workspace.
        dataset = ctx.find_asset("toy-dataset")
        if dataset is not None:
            ctx.artifact.save("dataset-path.txt", str(dataset.path))

    # 6. Catalog queries — flat view over the whole workspace.
    catalog = ws.catalog
    all_assets = catalog.query_assets()
    print(f"workspace root: {root}")
    print(f"total assets:   {len(all_assets)}")
    for asset in all_assets:
        kind = type(asset).__name__.removesuffix("Asset").lower()
        print(f"  [{kind:<11}] {asset.name:<20} scope={asset.scope.kind}")


if __name__ == "__main__":
    asyncio.run(main())
