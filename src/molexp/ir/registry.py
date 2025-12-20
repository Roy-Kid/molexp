from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


class OperationDefinition(BaseModel):
    op_id: str
    handler: Any  # Callable or Class
    schema_model: Type[BaseModel]
    version: str = "1.0"

    class Config:
        arbitrary_types_allowed = True


class OperationRegistry:
    def __init__(self):
        self._ops: Dict[str, OperationDefinition] = {}

    def register(self, op_id: str, schema: Type[BaseModel], version: str = "1.0"):
        """
        Decorator to register an operation.

        Usage:
            @registry.register("my.op", MyConfigModel)
            class MyTask(Task): ...
        """

        def decorator(handler: Any):
            self._ops[op_id] = OperationDefinition(
                op_id=op_id, handler=handler, schema_model=schema, version=version
            )
            return handler

        return decorator

    def get_operation(self, op_id: str) -> OperationDefinition:
        if op_id not in self._ops:
            raise ValueError(f"Operation '{op_id}' not found in registry.")
        return self._ops[op_id]

    def list_operations(self) -> Dict[str, OperationDefinition]:
        return self._ops.copy()


# Global instance
registry = OperationRegistry()
