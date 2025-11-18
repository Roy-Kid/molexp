"""Mock scientific workflow example."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from taskflow.engine import TaskEngine
from taskflow.task_base import EmptyConfig, Task


class Frame(BaseModel):
    id: int
    atoms: int


class LoadFrameTask(Task[EmptyConfig, list[Frame]]):
    cfg_model = EmptyConfig

    def forward(self, cfg: EmptyConfig) -> list[Frame]:  # type: ignore[override]
        return [Frame(id=1, atoms=100), Frame(id=2, atoms=120)]


class NeighborCfg(BaseModel):
    cutoff: float = 3.5


class ComputeNeighborsTask(Task[NeighborCfg, dict[int, int]]):
    cfg_model = NeighborCfg

    def forward(self, frames: list[Frame], cfg: NeighborCfg) -> dict[int, int]:
        return {frame.id: int(frame.atoms / cfg.cutoff) for frame in frames}


class RDFCfg(BaseModel):
    bins: int = 16


class ComputeRDFTask(Task[RDFCfg, list[float]]):
    cfg_model = RDFCfg

    def forward(self, neighbors: dict[int, int], cfg: RDFCfg) -> list[float]:
        scale = sum(neighbors.values()) or 1
        return [scale / cfg.bins for _ in range(cfg.bins)]


class ReportTask(Task[EmptyConfig, dict[str, Any]]):
    cfg_model = EmptyConfig

    def forward(self, frames: list[Frame], rdf: list[float], cfg: EmptyConfig) -> dict[str, Any]:
        return {"frames": len(frames), "rdf_mean": sum(rdf) / len(rdf)}


def build_pipeline() -> Task[Any, Any]:
    load = LoadFrameTask(name="frames")
    neighbors = ComputeNeighborsTask(load, name="neighbors")
    rdf = ComputeRDFTask(neighbors, name="rdf")
    report = ReportTask(load, rdf, name="report")
    return report


def run() -> None:
    engine = TaskEngine()
    pipeline = build_pipeline()
    result = engine.run(pipeline, cfg_overrides={"ComputeNeighborsTask__cutoff": 2.5, "bins": 8})
    print("Report:", result)


if __name__ == "__main__":
    run()
