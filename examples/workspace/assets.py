"""Artifacts, logs, checkpoints, data imports, and catalog queries.

Matches ``docs/guide/assets.md``.

Walks through:

1. ``ctx.artifact.save`` — writes a file and registers an ``ArtifactAsset``.
2. ``ctx.log(name).append`` — appends to a ``LogAsset`` scoped to the run.
3. ``ctx.checkpoint`` — writes a ``CheckpointAsset`` with parent chaining.
4. ``ws.data_assets.import_asset`` — pulls external data into the workspace.
5. ``ctx.find_asset`` — scope-walking lookup from inside a task.
6. ``ws.catalog.query_assets`` — workspace-wide asset queries.

Run directly::

    python examples/workspace/assets.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import promote_callable, Workflow


async def train(ctx: me.RunContext) -> None:
    # 1. Artifact — arbitrary payload written to run_dir/artifacts/.
    ctx.artifact.save("metrics.json", {"loss": 0.08, "acc": 0.94})

    # 2. Log — line-oriented, appendable, scoped to this run.
    log = ctx.log("train")
    log.append("epoch 1 start")
    log.append("epoch 1 done  loss=0.10")
    log.append("epoch 2 done  loss=0.08")

    # 3. Checkpoints — chained via parent_ckpt_id.
    ctx.checkpoint("epoch-1", data={"step": 1})
    ctx.checkpoint("epoch-2", data={"step": 2})

    # 5. find_asset walks experiment → project → workspace.
    dataset = ctx.find_asset("toy-dataset")
    if dataset is not None:
        ctx.artifact.save("dataset-path.txt", str(dataset.path))


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-assets-"))
    ws = me.Workspace(root, name="assets-demo")

    # 4. Import a workspace-scoped dataset from outside.
    external = root / "external.csv"
    external.write_text("x,y\n1,2\n3,4\n")
    ws.data_assets.import_asset("toy-dataset", external)

    project = ws.Project("demo")
    exp = project.Experiment("train")
    spec = promote_callable(train, name="train")
    spec.bind_to(exp)

    run = exp.Run()
    with run.start() as ctx:
        await spec.execute(run_context=ctx)

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
