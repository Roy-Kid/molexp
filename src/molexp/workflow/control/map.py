"""Map node: Apply operation to each element of a collection."""

from __future__ import annotations

from typing import Any, Iterable, List

from pydantic import BaseModel

from ..config import EmptyConfig
from ..node import Node
from ..plugin.registry import register


class MapConfig(EmptyConfig):
    """Configuration for MapNode.

    MapNode doesn't require configuration - the operation to apply
    is determined by the base_task passed during initialization.
    """

    pass


@register("control.map")
class MapNode(Node[MapConfig, List[Any]]):
    """Apply a node to each element of a collection.

    This is a fundamental control flow node that enables iteration
    over collections in workflows.

    Examples:
        >>> # In workflow context
        >>> base_task = SomeTransformNode(param="value")
        >>> map_node = MapNode(collection, base_task=base_task)
        >>> results = map_node(collection_data)
    """

    config_type = MapConfig

    def __init__(
        self,
        collection: Any,
        base_task: Node,
        id: str | None = None,
    ):
        """Initialize map node.

        Args:
            collection: Upstream node or constant providing the collection
            base_task: Node to apply to each element
            id: Unique identifier
        """
        super().__init__(collection, id=id or f"{base_task.id}__map")
        self.base_task = base_task

    def execute(self, collection: Iterable[Any]) -> List[Any]:
        """Apply base_task to each element.

        Args:
            collection: Collection to iterate over

        Returns:
            List of results
        """
        results: List[Any] = []
        for item in collection:
            if isinstance(item, tuple):
                result = self.base_task(*item)
            else:
                result = self.base_task(item)
            results.append(result)
        return results
