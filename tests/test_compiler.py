from __future__ import annotations

from pydantic import BaseModel

from molexp.compiler import compile_graph
from molexp.task_base import EmptyConfig, Task


class SourceTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> int:  # type: ignore[override]
        return 1


class SumCfg(BaseModel):
    bias: int = 0


class SumTask(Task[SumCfg, int]):
    cfg_model = SumCfg

    def forward(self, left: int, right: int, cfg: SumCfg) -> int:
        return left + right + cfg.bias


def test_compile_order() -> None:
    left = SourceTask(name="left")
    right = SourceTask(name="right")
    total = SumTask(left, right, name="sum")
    graph = compile_graph(total)
    assert graph.order[-1] is total
    assert left in graph.order and right in graph.order


def test_cycle_detection() -> None:
    node = SourceTask(name="cycle")
    node.upstreams.append(node)
    try:
        compile_graph(node)
    except ValueError as exc:  # pragma: no cover - error path
        assert "Cycle" in str(exc)
    else:
        raise AssertionError("Cycle not detected")
