from __future__ import annotations

from pydantic import BaseModel

from taskflow.compiler import compile_graph
from taskflow.engine import TaskEngine
from taskflow.task_base import EmptyConfig, Task


class SeedTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> int:  # type: ignore[override]
        return 2


class ScaleCfg(BaseModel):
    factor: int = 2
    offset: int = 0


class ScaleTask(Task[ScaleCfg, int]):
    cfg_model = ScaleCfg

    def forward(self, value: int, cfg: ScaleCfg) -> int:
        return value * cfg.factor + cfg.offset


def test_engine_executes_graph() -> None:
    source = SeedTask(name="seed")
    scale = ScaleTask(source, name="scale")
    engine = TaskEngine()
    result = engine.run(scale, cfg_overrides={"factor": 3, "scale__offset": 5})
    assert result == 11


def test_run_compiled_reuse() -> None:
    source = SeedTask(name="seed")
    scale = ScaleTask(source, name="scale")
    engine = TaskEngine()
    graph = compile_graph(scale)
    first = engine.run_compiled(graph)
    second = engine.run_compiled(graph, cfg_overrides={"factor": 4})
    assert first == 4
    assert second == 8
