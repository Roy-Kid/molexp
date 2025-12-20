"""Test the new unified node system with static configuration."""

import pytest
from pydantic import BaseModel, Field

from molexp.workflow.control import MapNode, ReduceNode
from molexp.workflow.node import Node
from molexp.workflow.plugin.registry import get_node_registry, register

# Get the global registry instance
node_registry = get_node_registry()


# Example nodes for testing


class DoubleConfig(BaseModel):
    """Config for doubling a number."""

    multiplier: int = 2


class DoubleNode(Node[DoubleConfig, int]):
    """Double a number."""

    config_type = DoubleConfig

    def execute(self, value: int) -> int:
        return value * self.config.multiplier


class SumConfig(BaseModel):
    """Config for summing numbers."""

    initial: float = 0.0


class SumNode(Node[SumConfig, float]):
    """Sum a list of numbers."""

    config_type = SumConfig

    def execute(self, *inputs: float) -> float:
        return sum(inputs) + self.config.initial


class ConstantConfig(BaseModel):
    """Config for constant generator."""

    value: int = Field(..., description="Constant value to return")


@register("test.constant")
class ConstantNode(Node[ConstantConfig, int]):
    """Generate a constant value."""

    config_type = ConstantConfig

    def execute(self) -> int:
        return self.config.value


def test_transform_node():
    """Test basic node with static config."""
    # Config is set at construction
    node = DoubleNode(multiplier=3)

    # Execute only receives input data
    result = node(5)
    assert result == 15

    # Test with default config
    node_default = DoubleNode()
    result = node_default(5)
    assert result == 10


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
    assert node_registry.has("test.constant")

    # Get registration
    reg = node_registry.get("test.constant")
    assert reg.node_class == ConstantNode
    assert reg.config_class == ConstantConfig


def test_map_node():
    """Test MapNode control flow."""
    # Create a collection
    numbers = [1, 2, 3, 4, 5]

    # Create base task with static config
    double_node = DoubleNode(multiplier=2)

    # Create map node
    map_node = MapNode(numbers, base_task=double_node)

    # Execute (map doesn't take config kwargs, uses EmptyConfig)
    result = map_node.execute(numbers)

    assert result == [2, 4, 6, 8, 10]


def test_reduce_node():
    """Test ReduceNode control flow."""
    numbers = [1, 2, 3, 4, 5]

    # Config at construction
    reduce_node = ReduceNode(numbers, method="sum")

    # Execute with data only
    result = reduce_node(numbers)
    assert result == 15

    # Test mean
    reduce_node_mean = ReduceNode(numbers, method="mean")
    result = reduce_node_mean(numbers)
    assert result == 3.0


def test_config_validation():
    """Test that config validation works at construction."""
    # Valid config
    node = DoubleNode(multiplier=3)
    result = node(5)
    assert result == 15

    # Invalid config should raise error at construction
    with pytest.raises(ValueError):
        node = DoubleNode(multiplier="not_a_number")


def test_config_serialization():
    """Test config serialization methods."""
    node = DoubleNode(multiplier=5)

    # Get config as dict
    config_dict = node.get_config_dict()
    assert config_dict == {"multiplier": 5}

    # Create node from config dict
    node2 = DoubleNode.from_config_dict(config_dict)
    assert node2.config.multiplier == 5
    assert node2(10) == 50


def test_node_repr():
    """Test node representation."""
    node = DoubleNode(multiplier=3, id="my_doubler")
    repr_str = repr(node)
    assert "DoubleNode" in repr_str
    assert "my_doubler" in repr_str
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
