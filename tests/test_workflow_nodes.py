"""Test the new unified node system."""

import pytest
from pydantic import BaseModel, Field

from molexp.workflow.node import Node
from molexp.workflow.primitives import TransformNode, AggregateNode, GeneratorNode
from molexp.workflow.control import MapNode, ReduceNode
from molexp.workflow.registry import register, node_registry


# Example nodes for testing

class DoubleConfig(BaseModel):
    """Config for doubling a number."""
    multiplier: int = 2


class DoubleNode(TransformNode[DoubleConfig, int, int]):
    """Double a number."""
    config_type = DoubleConfig
    
    def transform(self, value: int, config: DoubleConfig) -> int:
        return value * config.multiplier


class SumConfig(BaseModel):
    """Config for summing numbers."""
    initial: float = 0.0


class SumNode(AggregateNode[SumConfig, float, float]):
    """Sum a list of numbers."""
    config_type = SumConfig
    
    def aggregate(self, inputs: list[float], config: SumConfig) -> float:
        return sum(inputs) + config.initial


class ConstantConfig(BaseModel):
    """Config for constant generator."""
    value: int = Field(..., description="Constant value to return")


@register("test.constant")
class ConstantNode(GeneratorNode[ConstantConfig, int]):
    """Generate a constant value."""
    config_type = ConstantConfig
    
    def generate(self, config: ConstantConfig) -> int:
        return config.value


def test_transform_node():
    """Test TransformNode primitive."""
    node = DoubleNode()
    
    # Test with default config
    result = node(5)
    assert result == 10
    
    # Test with custom config
    result = node(5, multiplier=3)
    assert result == 15


def test_generator_node():
    """Test GeneratorNode primitive."""
    node = ConstantNode()
    
    # Test generation
    result = node(value=42)
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
    
    # Create base task
    double_node = DoubleNode()
    
    # Create map node
    map_node = MapNode(numbers, base_task=double_node)
    
    # Execute (map doesn't take config kwargs, uses EmptyConfig)
    result = map_node.execute(numbers, map_node.config_type())
    
    assert result == [2, 4, 6, 8, 10]


def test_reduce_node():
    """Test ReduceNode control flow."""
    numbers = [1, 2, 3, 4, 5]
    
    reduce_node = ReduceNode(numbers)
    
    # Test sum
    result = reduce_node(numbers, method="sum")
    assert result == 15
    
    # Test mean
    result = reduce_node(numbers, method="mean")
    assert result == 3.0


def test_config_validation():
    """Test that config validation works."""
    node = DoubleNode()
    
    # Valid config
    result = node(5, multiplier=3)
    assert result == 15
    
    # Invalid config should raise error
    with pytest.raises(ValueError):
        node(5, multiplier="not_a_number")


if __name__ == "__main__":
    # Run tests
    test_transform_node()
    test_generator_node()
    test_node_registration()
    test_map_node()
    test_reduce_node()
    test_config_validation()
    
    print("✓ All tests passed!")
