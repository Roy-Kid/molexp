# molexp Documentation

## Concepts

molexp is organized around a strict separation of concerns:

1. **Task** – a pure computation node backed by a `forward` method and Pydantic config.
2. **Compiler** – statically walks the graph and produces a deterministic execution order.
3. **Engine** – executes a compiled graph, performing config override merges on the fly.
4. **DSL** – constructs structural nodes such as map/reduce/if-else/repeat for ergonomic flows.

## Quickstart

```python
from molexp.task_base import Task, EmptyConfig
from molexp.engine import TaskEngine

class AddOne(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig
    out_model = None

    def forward(self, value: int, cfg: EmptyConfig) -> int:
        return value + 1

load = AddOne(name="add1")
engine = TaskEngine()
result = engine.run(load)
```

## Graph Compilation

Use `molexp.compiler.compile_graph` to obtain a `CompiledGraph`. The compiler performs DFS
traversal, detects cycles, and ensures deterministic ordering.

## Engine

`TaskEngine.run(root)` compiles the graph and executes it. To pass runtime overrides use a mapping:
`{"scale": 2, "MultiplyTask__factor": 4}`. Node-specific overrides use the `TaskName__field`
pattern.

## DSL Examples

```python
load = LoadArrayTask(name="load")
mapped = multiply_task.map(load)
reduced = mapped.reduce("mean")
branch = predicate_task.if_else(mapped, reduced)
looped = multiply_task.repeat(5)
```

See `src/molexp/examples/` for full runnable samples.
