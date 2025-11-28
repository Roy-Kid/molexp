# molexp

molexp is a tiny yet fully-typed task-graph framework built on top of Pydantic. It contains a
pure functional task abstraction, a static compiler that produces deterministic graph orders, a
runtime engine, and a tiny DSL for common data-flow patterns. This repository is intentionally
minimal to highlight how each layer works without hidden magic.

```
+-----------+       +-----------+       +---------+
|   Task    |  -->  | Compiler  |  -->  | Engine  |
+-----------+       +-----------+       +---------+
        ^                  |                  |
        |                  v                  v
        +----------- DSL abstractions --------+
```

## Quick Example

```python
from molexp.task_base import Task, EmptyConfig
from molexp.engine import TaskEngine

class MultiplyTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig
    out_model = None

    def forward(self, value: int, cfg: EmptyConfig) -> int:
        return value * 2

mult = MultiplyTask(name="multiply")
engine = TaskEngine()
result = engine.run(mult)
print(result)
```

See the [docs](./docs/README.md) for an in-depth tour of the architecture, compiler, engine, and DSL
usage.
