"""Internal node implementations for the DSL."""

from __future__ import annotations

from typing import Any, Iterable

from ..task_base import EmptyConfig, Task


class MapNode(Task[EmptyConfig, list[Any]]):
    """Applies a base task to each item of a collection."""

    cfg_model = EmptyConfig

    def __init__(self, base_task: Task[Any, Any], collection: Any) -> None:
        super().__init__(collection, name=f"{base_task.name}__map")
        self.base_task = base_task

    def forward(self, collection: Iterable[Any], cfg: EmptyConfig) -> list[Any]:  # noqa: D401
        results: list[Any] = []
        for item in collection:
            if isinstance(item, tuple):
                results.append(self.base_task(*item))
            else:
                results.append(self.base_task(item))
        return results


class ReduceNode(Task[EmptyConfig, Any]):
    """Reduces an iterable using a named strategy."""

    cfg_model = EmptyConfig

    def __init__(self, source: Task[Any, Iterable[Any]], method: str) -> None:
        super().__init__(source, name=f"reduce_{method}")
        self.method = method

    def forward(self, iterable: Iterable[Any], cfg: EmptyConfig) -> Any:  # noqa: D401
        data = list(iterable)
        if self.method == "sum":
            return sum(data)
        if self.method == "mean":
            return sum(data) / len(data) if data else 0
        if self.method == "max":
            return max(data)
        if self.method == "min":
            return min(data)
        raise ValueError(f"Unknown reduce method {self.method}")


class IfElseNode(Task[EmptyConfig, Any]):
    """Static branch selecting between pre-computed branches."""

    cfg_model = EmptyConfig

    def __init__(self, cond: Task[Any, bool], then_task: Task[Any, Any], else_task: Task[Any, Any]) -> None:
        super().__init__(cond, then_task, else_task, name=f"{cond.name}__ifelse")

    def forward(self, condition: bool, then_value: Any, else_value: Any, cfg: EmptyConfig) -> Any:  # noqa: D401
        return then_value if condition else else_value


class RepeatNode(Task[EmptyConfig, Any]):
    """Applies ``base_task`` to the result of the previous iteration."""

    cfg_model = EmptyConfig

    def __init__(self, base_task: Task[Any, Any], upstream: Task[Any, Any], iteration: int) -> None:
        super().__init__(upstream, name=f"{base_task.name}__repeat_{iteration}")
        self.base_task = base_task

    def forward(self, value: Any, cfg: EmptyConfig) -> Any:  # noqa: D401
        if isinstance(value, tuple):
            return self.base_task(*value)
        return self.base_task(value)
