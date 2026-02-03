"""Builtin nodes registration entry point."""

from molexp.workflow.plugin.registry import TaskRegistry

# Import control tasks to trigger registration
from molexp.workflow.control import conditional, loop, map, reduce

def register_builtin_nodes(registry: TaskRegistry) -> None:
    """Register builtin nodes.
    
    This function is called by the plugin loader.
    Simply importing the control modules above is sufficient to register
    default tasks via their @register decorators to the global registry.
    """
    pass
