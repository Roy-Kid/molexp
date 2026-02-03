"""Test the new unified node system with static configuration."""

import pytest
from pydantic import BaseModel, Field

from molexp.workflow.control import MapTask, ReduceTask
from molexp.workflow.task import Task
from molexp.workflow.plugin.registry import get_task_registry, register

# Get the global registry instance
task_registry = get_task_registry()


# Example tasks for testing


class DoubleConfig(BaseModel):
    """Config for doubling a number."""

    multiplier: int = 2


class DoubleNode(Task[DoubleConfig, int]):
    """Double a number."""

    config_type = DoubleConfig
    inputs = {"value": int}
    outputs = {"result": int}

    def execute(self, ctx=None, **inputs) -> dict[str, int]:
        value = inputs.get("value", 0)
        return {"result": value * self.config.multiplier}


class SumConfig(BaseModel):
    """Config for summing numbers."""

    initial: float = 0.0


class SumNode(Task[SumConfig, float]):
    """Sum a list of numbers."""

    config_type = SumConfig

    def execute(self, *inputs: float) -> float:
        return sum(inputs) + self.config.initial


class ConstantConfig(BaseModel):
    """Config for constant generator."""

    value: int = Field(..., description="Constant value to return")


@register("test.constant")
class ConstantNode(Task[ConstantConfig, int]):
    """Generate a constant value."""

    config_type = ConstantConfig

    def execute(self, ctx=None, **inputs) -> int:
        return self.config.value


def test_transform_node():
    """Test basic node with static config."""
    # Config is set at construction
    node = DoubleNode(multiplier=3)

    # Execute only receives input data
    result = node(value=5)
    assert result == {"result": 15}

    # Test with default config
    node_default = DoubleNode()
    result = node_default(value=5)
    assert result == {"result": 10}


def test_generator_node():
    """Test generator node (no inputs)."""
    # Config at construction
    node = ConstantNode(value=42)

    # Execute with no arguments
    result = node()
    assert result == 42


def test_node_registration():
    """Test node registry."""
    # Check that ConstantNode was registered
    assert task_registry.has("test.constant")

    # Get registration
    reg = task_registry.get("test.constant")
    assert reg.task_class == ConstantNode
    assert reg.config_class == ConstantConfig


def test_map_node():
    """Test MapTask control flow."""
    # Create a collection
    numbers = [1, 2, 3, 4, 5]

    # Create base task with static config
    double_node = DoubleNode(multiplier=2)

    # Create map task (no runtime data in __init__)
    map_task = MapTask(double_node, map_over="items")

    # Execute with collection via **inputs
    result = map_task.execute(ctx=None, items=numbers)

    assert result == {"results": [{"result": 2}, {"result": 4}, {"result": 6}, {"result": 8}, {"result": 10}]}


def test_reduce_node():
    """Test ReduceTask control flow."""
    numbers = [1, 2, 3, 4, 5]

    # Config at construction (method only)
    reduce_task = ReduceTask(method="sum")

    # Execute with data via **inputs
    result = reduce_task.execute(ctx=None, collection=numbers)
    assert result == {"result": 15}

    # Test mean
    reduce_task_mean = ReduceTask(method="mean")
    result = reduce_task_mean.execute(ctx=None, collection=numbers)
    assert result == {"result": 3.0}


def test_config_validation():
    """Test that config validation works at construction."""
    # Valid config
    node = DoubleNode(multiplier=3)
    result = node(value=5)
    assert result == {"result": 15}

    # Invalid config should raise error at construction
    with pytest.raises(ValueError):
        node = DoubleNode(multiplier="not_a_number")


def test_config_serialization():
    """Test config serialization methods."""
    node = DoubleNode(multiplier=5)

    # Get config as dict using dump()
    config_dict = node.dump()
    assert config_dict == {"multiplier": 5}

    # Create node from config dict
    node2 = DoubleNode(**config_dict)
    assert node2.config.multiplier == 5
    result = node2(value=10)
    assert result == {"result": 50}


def test_node_repr():
    """Test node representation."""
    node = DoubleNode(multiplier=3)
    repr_str = repr(node)
    assert "DoubleNode" in repr_str
    assert "multiplier=3" in repr_str


if __name__ == "__main__":
    # Run tests
    test_transform_node()
    test_generator_node()
    test_node_registration()
    test_map_node()
    test_reduce_node()
    test_config_validation()
    test_config_serialization()
    test_node_repr()

    print("✓ All tests passed!")
