"""Tests for protocol interface and strict registry enforcement."""

import pytest
from typing import Any

from molexp.workflow import TaskProtocol, Task, register_task, get_task_class, get_task_id


# Test fixtures: Mock Compute class (simulating molpy.Compute)
class MockCompute:
    """Mock Compute class that implements TaskProtocol without inheriting from Task."""
    
    input_key: str = "input"
    output_key: str = "result"
    
    def __init__(self, param1: str, param2: int, **config_kwargs: Any):
        self._config = {"param1": param1, "param2": param2, **config_kwargs}
        self.param1 = param1
        self.param2 = param2
    
    def execute(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Execute as workflow task."""
        input_value = inputs.get(self.input_key)
        if input_value is None:
            raise ValueError(f"Missing required input: {self.input_key}")
        
        result = self._compute(input_value)
        return {self.output_key: result}
    
    def _compute(self, input_value: Any) -> Any:
        """Core computation logic."""
        return f"{self.param1}_{input_value}_{self.param2}"
    
    def dump(self) -> dict[str, Any]:
        """Serialize configuration."""
        return self._config.copy()


class TestProtocolCompatibility:
    """Test TaskProtocol structural typing."""
    
    def test_protocol_instance_check(self):
        """Test that MockCompute is recognized as TaskProtocol."""
        compute = MockCompute(param1="test", param2=42)
        assert isinstance(compute, TaskProtocol)
    
    def test_protocol_missing_execute(self):
        """Test that class without execute() is not TaskProtocol."""
        class BadCompute:
            def dump(self) -> dict[str, Any]:
                return {}
        
        bad = BadCompute()
        assert not isinstance(bad, TaskProtocol)
    
    def test_protocol_missing_dump(self):
        """Test that class without dump() is not TaskProtocol."""
        class BadCompute:
            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}
        
        bad = BadCompute()
        assert not isinstance(bad, TaskProtocol)


class TestComputeAsTask:
    """Test using Compute classes as workflow tasks."""
    
    def test_compute_execute(self):
        """Test executing Compute via execute() method."""
        compute = MockCompute(param1="hello", param2=99)
        result = compute.execute(input="world")
        
        assert result == {"result": "hello_world_99"}
    
    def test_compute_dump(self):
        """Test serializing Compute config."""
        compute = MockCompute(param1="test", param2=42)
        config = compute.dump()
        
        assert config == {"param1": "test", "param2": 42}
    
    def test_compute_custom_keys(self):
        """Test Compute with custom input/output keys."""
        class CustomCompute(MockCompute):
            input_key = "data"
            output_key = "output"
        
        compute = CustomCompute(param1="custom", param2=1)
        result = compute.execute(data="test")
        
        assert result == {"output": "custom_test_1"}
    
    def test_compute_missing_input(self):
        """Test error when required input is missing."""
        compute = MockCompute(param1="test", param2=1)
        
        with pytest.raises(ValueError, match="Missing required input: input"):
            compute.execute(wrong_key="value")


class TestTaskConfigTypes:
    """Test Task with Pydantic-only config types."""

    def test_task_with_non_pydantic_config_rejected(self):
        """Task must enforce Pydantic BaseModel configs."""
        class BadConfig:
            pass

        class BadTask(Task[BadConfig, dict]):
            config_type = BadConfig

            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {"result": "bad"}

        with pytest.raises(ValueError, match="config_type must be a Pydantic BaseModel"):
            BadTask()


class TestRegistration:
    """Test registering protocol-compatible classes."""
    
    def test_register_compute_class(self):
        """Test registering a non-Task class is rejected."""
        with pytest.raises(ValueError, match="config_type"):
            register_task(MockCompute)

    def test_register_class_without_execute(self):
        """Test that registering class without execute() raises error."""
        class BadTask:
            def dump(self) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="config_type"):
            register_task(BadTask)

    def test_register_class_without_dump(self):
        """Test that registering class without dump() raises error."""
        class BadTask:
            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="config_type"):
            register_task(BadTask)

    def test_register_task_with_deterministic_id(self):
        """Test deterministic task ID derivation for Task classes."""
        from pydantic import BaseModel

        class MyConfig(BaseModel):
            value: int = 1

        class MyTask(Task[MyConfig, dict]):
            config_type = MyConfig

            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {"result": self.config.value}

        task_type_id = register_task(MyTask)
        assert task_type_id == f"{MyTask.__module__}.{MyTask.__qualname__}"
        assert get_task_id(MyTask) == task_type_id
        assert get_task_class(task_type_id) is MyTask


class TestMixedWorkflow:
    """Test workflows mixing Pydantic Tasks and Compute tasks."""
    
    def test_mixed_task_types(self):
        """Test that Pydantic Tasks execute as expected."""
        from pydantic import BaseModel
        
        # Pydantic-based Task
        class PydanticConfig(BaseModel):
            multiplier: int
        
        class PydanticTask(Task[PydanticConfig, dict]):
            config_type = PydanticConfig
            
            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                value = inputs.get("value", 1)
                return {"result": value * self.config.multiplier}
        
        # Create instances
        pydantic_task = PydanticTask(multiplier=2)
        # Should be TaskProtocol
        assert isinstance(pydantic_task, TaskProtocol)

        # Should execute
        result1 = pydantic_task.execute(value=5)
        assert result1 == {"result": 10}

        # Should dump
        config1 = pydantic_task.dump()
        assert config1 == {"multiplier": 2}


class TestConfigSerialization:
    """Test config serialization for Pydantic-only configs."""
    
    def test_pydantic_dump(self):
        """Test dumping Pydantic config."""
        from pydantic import BaseModel
        
        class Config(BaseModel):
            name: str
            value: int
        
        class MyTask(Task[Config, dict]):
            config_type = Config
            
            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}
        
        task = MyTask(name="test", value=42)
        dumped = task.dump()
        
        assert dumped == {"name": "test", "value": 42}
    
    def test_dump_rejects_non_pydantic(self):
        """Non-Pydantic configs should be rejected at construction."""
        class BadConfig:
            pass

        class BadTask(Task[BadConfig, dict]):
            config_type = BadConfig

            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="config_type must be a Pydantic BaseModel"):
            BadTask()


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_pydantic_validation_error(self):
        """Test that Pydantic validation errors are caught."""
        from pydantic import BaseModel
        
        class Config(BaseModel):
            value: int
        
        class MyTask(Task[Config, dict]):
            config_type = Config
            
            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}
        
        with pytest.raises(ValueError, match="Invalid Pydantic config"):
            MyTask(value="not_an_int")
    
    def test_non_pydantic_config_rejected(self):
        """Non-Pydantic configs are rejected before execution."""
        class BadConfig:
            value: int

        class MyTask(Task[BadConfig, dict]):
            config_type = BadConfig

            def execute(self, ctx=None, **inputs: Any) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="config_type must be a Pydantic BaseModel"):
            MyTask(value=1)
