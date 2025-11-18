"""Numeric pipeline example demonstrating the DSL."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from taskflow.engine import TaskEngine
from taskflow.task_base import EmptyConfig, Task


class LoadArrayTask(Task[EmptyConfig, list[int]]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> list[int]:  # type: ignore[override]
        return [1, 2, 3, 4]


class AddCfg(BaseModel):
    addend: int = 1


class AddTask(Task[AddCfg, int]):
    cfg_model = AddCfg

    def forward(self, value: int, cfg: AddCfg) -> int:
        return value + cfg.addend


class MultiplyCfg(BaseModel):
    factor: int = 2


class MultiplyTask(Task[MultiplyCfg, int]):
    cfg_model = MultiplyCfg

    def forward(self, value: int, cfg: MultiplyCfg) -> int:
        return value * cfg.factor


class ThresholdCfg(BaseModel):
    threshold: int = 10


class ThresholdTask(Task[ThresholdCfg, bool]):
    cfg_model = ThresholdCfg

    def forward(self, value: int, cfg: ThresholdCfg) -> bool:
        return value > cfg.threshold


class SeedTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> int:  # type: ignore[override]
        return 2


class BundleTask(Task[EmptyConfig, dict[str, Any]]):
    cfg_model = EmptyConfig

    def forward(self, branch_value: Any, repeated_value: Any, cfg: EmptyConfig) -> dict[str, Any]:
        return {"branch": branch_value, "loop": repeated_value}


def build_pipeline() -> Task[Any, Any]:
    load = LoadArrayTask(name="load")
    add = AddTask(name="add")
    added = add.map(load)
    multiplied = MultiplyTask(name="mult").map(load)
    summed = added.reduce("sum")
    mean_value = multiplied.reduce("mean")
    cond = ThresholdTask(summed, name="threshold")
    branch = cond.if_else(summed, mean_value)
    seed = SeedTask(name="seed")
    scale = MultiplyTask(seed, name="scale_seed")
    repeated = scale.repeat(3)
    bundle = BundleTask(branch, repeated, name="bundle")
    return bundle


def run() -> None:
    engine = TaskEngine()
    pipeline = build_pipeline()
    result = engine.run(pipeline)
    print("Pipeline result:", result)


if __name__ == "__main__":
    run()
