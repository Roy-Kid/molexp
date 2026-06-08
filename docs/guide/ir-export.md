# Which IR export do I use?

A `CompiledWorkflow` offers two JSON/graph exports. They are **not**
interchangeable — pick by what you need.

| You need… | Use | Why |
|---|---|---|
| A UI canvas, control-flow visualization, observability — the **full graph** | `to_graph_ir()` | Total; emits tasks, deps, entries, control, branch, loops, **and** the parallel fan-out edges (`map_over→body`, `body→join`, tagged `kind="parallel"`). Never requires a `task_type` slug. |
| Persistence, round-trip (`from_ir`), or script generation (`to_python`) — the **wire format** | `to_ir()` | The data-DAG round-trippable form. Requires a `task_type` slug per task under `strict=True`, and **omits** the parallel fan-out / control topology by design. |

The same split exists for diagrams: `to_mermaid()` renders the data DAG,
`to_graph_mermaid()` renders the full graph.

## The asymmetry to know

`to_ir()` is the *data DAG*: it carries data-dependency edges only. The
parallel fan-out edges (`map_over → body` and `body → join`) and other
control/branch/loop topology live in the **full graph** and are emitted only by
`to_graph_ir()`. If you reach for `to_ir()` to drive a UI and find the parallel
edges missing, that is by design — switch to `to_graph_ir()`; you do not need to
hand-patch the IR to inject them.

```python
compiled = wf.compile()

graph = compiled.to_graph_ir()          # full graph (UI / observability)
parallel_edges = [e for e in graph.edges if e.kind == "parallel"]

wire = compiled.to_ir()                 # round-trippable wire format
rebuilt = CompiledWorkflow.from_ir(wire)
```
