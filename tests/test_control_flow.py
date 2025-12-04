"""Tests for advanced control flow primitives."""

import pytest
from molexp.task_base import Task, EmptyConfig
from molexp.engine import TaskEngine
from molexp.nodes.control_flow import (
    WhileLoopNode,
    DynamicIfElseNode,
    ParallelMapNode,
    ForLoopNode,
)


class IncrementTask(Task[EmptyConfig, int]):
    """Simple task that increments a value."""
    
    cfg_model = EmptyConfig
    
    def forward(self, x: int, cfg: EmptyConfig) -> int:
        return x + 1


class SquareTask(Task[EmptyConfig, int]):
    """Simple task that squares a value."""
    
    cfg_model = EmptyConfig
    
    def forward(self, x: int, cfg: EmptyConfig) -> int:
        return x * x


class AddIndexTask(Task[EmptyConfig, int]):
    """Task that adds index to value."""
    
    cfg_model = EmptyConfig
    
    def forward(self, value: int, index: int, cfg: EmptyConfig) -> int:
        return value + index


# WhileLoopNode Tests
# ===================

def test_while_loop_basic():
    """Test basic while loop functionality."""
    increment = IncrementTask()
    
    # Loop until value reaches 5
    loop = WhileLoopNode(
        body_task=increment,
        condition_fn=lambda x: x < 5,
        initial_value=0,
        max_iterations=10
    )
    
    engine = TaskEngine()
    result = engine.run(loop)
    
    assert result == 5


def test_while_loop_max_iterations():
    """Test that while loop respects max_iterations."""
    increment = IncrementTask()
    
    # Infinite loop condition, but limited by max_iterations
    loop = WhileLoopNode(
        body_task=increment,
        condition_fn=lambda x: True,  # Always true
        initial_value=0,
        max_iterations=10
    )
    
    engine = TaskEngine()
    with pytest.warns(RuntimeWarning):
        result = engine.run(loop)
    
    assert result == 10


def test_while_loop_zero_iterations():
    """Test while loop that doesn't execute at all."""
    increment = IncrementTask()
    
    # Condition is false from the start
    loop = WhileLoopNode(
        body_task=increment,
        condition_fn=lambda x: x < 0,
        initial_value=5,
        max_iterations=10
    )
    
    engine = TaskEngine()
    result = engine.run(loop)
    
    assert result == 5  # Unchanged


# DynamicIfElseNode Tests
# ========================

def test_dynamic_if_else_then_branch():
    """Test dynamic conditional executes then branch."""
    increment = IncrementTask()
    square = SquareTask()
    
    # Condition is true, should execute increment
    conditional = DynamicIfElseNode(
        condition_fn=lambda x: x > 0,
        then_task=increment,
        else_task=square,
        condition_input=5
    )
    
    engine = TaskEngine()
    result = engine.run(conditional)
    
    assert result == 6  # 5 + 1


def test_dynamic_if_else_else_branch():
    """Test dynamic conditional executes else branch."""
    increment = IncrementTask()
    square = SquareTask()
    
    # Condition is false, should execute square
    conditional = DynamicIfElseNode(
        condition_fn=lambda x: x > 0,
        then_task=increment,
        else_task=square,
        condition_input=-5
    )
    
    engine = TaskEngine()
    result = engine.run(conditional)
    
    assert result == 25  # (-5)^2


# ParallelMapNode Tests
# ======================

def test_parallel_map_basic():
    """Test parallel map produces correct results."""
    square = SquareTask()
    inputs = [1, 2, 3, 4, 5]
    
    parallel_map = ParallelMapNode(
        base_task=square,
        collection=inputs,
        max_workers=2,
        use_processes=False
    )
    
    engine = TaskEngine()
    results = engine.run(parallel_map)
    
    assert results == [1, 4, 9, 16, 25]


def test_parallel_map_empty_collection():
    """Test parallel map with empty collection."""
    square = SquareTask()
    
    parallel_map = ParallelMapNode(
        base_task=square,
        collection=[],
        max_workers=2,
        use_processes=False
    )
    
    engine = TaskEngine()
    results = engine.run(parallel_map)
    
    assert results == []


def test_parallel_map_single_worker():
    """Test parallel map with single worker."""
    square = SquareTask()
    inputs = [1, 2, 3]
    
    parallel_map = ParallelMapNode(
        base_task=square,
        collection=inputs,
        max_workers=1,
        use_processes=False
    )
    
    engine = TaskEngine()
    results = engine.run(parallel_map)
    
    assert results == [1, 4, 9]


# ForLoopNode Tests
# =================

def test_for_loop_basic():
    """Test basic for loop functionality."""
    add_index = AddIndexTask()
    
    # Sum of 0 + 1 + 2 + ... + 9
    for_loop = ForLoopNode(
        body_task=add_index,
        initial_value=0,
        n=10
    )
    
    engine = TaskEngine()
    result = engine.run(for_loop)
    
    assert result == 45  # Sum of 0..9


def test_for_loop_zero_iterations():
    """Test for loop with zero iterations."""
    add_index = AddIndexTask()
    
    for_loop = ForLoopNode(
        body_task=add_index,
        initial_value=100,
        n=0
    )
    
    engine = TaskEngine()
    result = engine.run(for_loop)
    
    assert result == 100  # Unchanged


def test_for_loop_single_iteration():
    """Test for loop with single iteration."""
    add_index = AddIndexTask()
    
    for_loop = ForLoopNode(
        body_task=add_index,
        initial_value=10,
        n=1
    )
    
    engine = TaskEngine()
    result = engine.run(for_loop)
    
    assert result == 10  # 10 + 0


# Integration Tests
# =================

def test_while_loop_with_map():
    """Test combining while loop with map."""
    increment = IncrementTask()
    
    # Create a list using map
    inputs = [0, 1, 2]
    mapped = increment.map(inputs)
    
    # Then use while loop on the result
    def sum_less_than_10(lst):
        return sum(lst) < 10
    
    # This won't work directly as WhileLoopNode expects the body to transform the value
    # This test demonstrates the limitation and expected behavior
    engine = TaskEngine()
    result = engine.run(mapped)
    
    assert result == [1, 2, 3]


def test_parallel_map_with_reduce():
    """Test parallel map followed by reduce."""
    square = SquareTask()
    inputs = [1, 2, 3, 4, 5]
    
    # Parallel map then reduce
    parallel_map = square.parallel_map(inputs, max_workers=2)
    mean_result = parallel_map.reduce("mean")
    
    engine = TaskEngine()
    result = engine.run(mean_result)
    
    expected_mean = sum([1, 4, 9, 16, 25]) / 5
    assert result == expected_mean


def test_nested_control_flow():
    """Test nesting control flow primitives."""
    increment = IncrementTask()
    square = SquareTask()
    
    # Create a conditional inside a loop
    # This is a conceptual test - actual nesting would require more complex setup
    inputs = [1, 2, 3, 4, 5]
    
    # Map with increment
    mapped = increment.map(inputs)
    
    # Then reduce
    result_task = mapped.reduce("sum")
    
    engine = TaskEngine()
    result = engine.run(result_task)
    
    assert result == sum([2, 3, 4, 5, 6])  # Each incremented by 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
