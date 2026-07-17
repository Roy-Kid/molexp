"""IR export: Mermaid diagrams, JSON IR round-trip, and graph IR for UIs.

Matches ``docs/guide/ir-export.md``.

Demonstrates:

1. ``to_mermaid()`` — data-DAG diagram (``flowchart LR``).
2. ``to_graph_mermaid()`` — full-graph diagram with control-flow topology.
3. ``to_ir()`` — wire format for persistence and ``CompiledWorkflow.from_ir()``.
4. ``to_graph_ir()`` — full graph IR for UI / observability consumers.

Run directly::

    python examples/workflow/ir_export.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import (
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
    default_registry,
)


@default_registry.register("docs.fetch")
class Fetch(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        return {"values": [1, 2, 3]}


@default_registry.register("docs.summarize")
class Summarize(Task):
    async def execute(self, ctx: TaskContext, values: list[int]) -> int:
        return sum(values)


async def main() -> None:
    compiled = (
        WorkflowCompiler(name="demo").add(Fetch()).add(Summarize(), depends_on=["fetch"]).compile()
    )

    # 1. Data-DAG Mermaid diagram (``flowchart LR``)
    print("── to_mermaid() ──────────────────────────────────────────")
    print(compiled.to_mermaid())
    print()

    # 2. Full-graph Mermaid — includes control-flow / parallel topology
    print("── to_graph_mermaid() ────────────────────────────────────")
    print(compiled.to_graph_mermaid())
    print()

    # 3. Wire-format IR + round-trip back to CompiledWorkflow
    wire = compiled.to_ir()
    rebuilt = compiled.from_ir(wire)
    result = await WorkflowRuntime().execute(rebuilt)
    print("IR round-trip status:", result.status)
    print("IR round-trip outputs:", result.outputs)
    print()

    # 4. Full graph IR (UI-facing — edges carry kind annotations)
    graph = compiled.to_graph_ir()
    print(f"Graph IR tasks:    {len(graph.tasks)}")
    print(f"Graph IR edges:    {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}  (kind={edge.kind})")


if __name__ == "__main__":
    asyncio.run(main())
