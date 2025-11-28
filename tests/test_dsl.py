from __future__ import annotations

from pydantic import BaseModel

from molexp.engine import TaskEngine
from molexp.task_base import EmptyConfig, Task


class ValueTask(Task[EmptyConfig, list[int]]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> list[int]:  # type: ignore[override]
        return [1, 2, 3]


class IncrementCfg(BaseModel):
    delta: int = 1


class IncrementTask(Task[IncrementCfg, int]):
    cfg_model = IncrementCfg

    def forward(self, value: int, cfg: IncrementCfg) -> int:
        return value + cfg.delta


class PredicateTask(Task[EmptyConfig, bool]):
    cfg_model = EmptyConfig

    def forward(self, values: list[int], cfg: EmptyConfig) -> bool:
        return sum(values) > 5


class SeedNumberTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> int:  # type: ignore[override]
        return 1


def test_map_reduce_ifelse() -> None:
    values = ValueTask(name="values")
    inc = IncrementTask(name="inc")
    mapped = inc.map(values)
    reduced = mapped.reduce("sum")
    predicate = PredicateTask(mapped, name="pred")
    branch = predicate.if_else(reduced, mapped.reduce("mean"))
    engine = TaskEngine()
    result = engine.run(branch)
    assert isinstance(result, (int, float))


def test_repeat_unroll() -> None:
    base = IncrementTask(SeedNumberTask(name="seed"), name="inc_base")
    repeated = base.repeat(3)
    engine = TaskEngine()
    result = engine.run(repeated)
    assert result == 4
