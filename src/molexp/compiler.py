"""Graph compiler producing deterministic execution order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

from .task_base import Task


@dataclass(slots=True)
class CompiledGraph:
    """Result of compiling a task graph."""

    root: Task
    order: list[Task]

    def __iter__(self) -> Iterable[Task]:
        return iter(self.order)


def compile_graph(root: Task) -> CompiledGraph:
    """Compile ``root`` into a ``CompiledGraph``.

    Parameters
    ----------
    root:
        Graph output node.

    Raises
    ------
    ValueError
        If cycles are detected.
    """

    order: list[Task] = []
    visited: Set[Task] = set()
    stack: Set[Task] = set()

    def dfs(node: Task) -> None:
        if node in stack:
            raise ValueError(f"Cycle detected at {node.name}")
        if node in visited:
            return
        stack.add(node)
        for upstream in node.iter_task_upstreams():
            dfs(upstream)
        stack.remove(node)
        visited.add(node)
        order.append(node)

    dfs(root)
    return CompiledGraph(root=root, order=order)
