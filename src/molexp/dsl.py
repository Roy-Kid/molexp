"""User-facing DSL helpers."""

from __future__ import annotations

from typing import Any

from .nodes import IfElseNode, MapNode, ReduceNode, RepeatNode
from .task_base import Task


def map_task(base: Task[Any, Any], collection: Any) -> Task[Any, list[Any]]:
    """Create a map node."""

    return MapNode(base, collection)


def reduce_task(source: Task[Any, Any], method: str) -> Task[Any, Any]:
    """Create a reduce node."""

    return ReduceNode(source, method)


def if_else_task(cond: Task[Any, bool], then_task: Task[Any, Any], else_task: Task[Any, Any]) -> Task[Any, Any]:
    """Create a branch node."""

    return IfElseNode(cond, then_task, else_task)


def repeat_task(base: Task[Any, Any], n: int) -> Task[Any, Any]:
    """Unroll ``n`` sequential applications of ``base``."""

    if n <= 1:
        return base
    node: Task[Any, Any] = base
    for idx in range(1, n):
        node = RepeatNode(base, node, idx)
    return node
