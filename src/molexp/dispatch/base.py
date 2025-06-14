from typing import Protocol
from molexp import Task

class TaskDispatcher(Protocol):
    def dispatch(self, task: Task, context: dict) -> object:
        ...
