"""Entry point discovery and plugin loading."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Callable

from .registry import NodeRegistry, get_node_registry

logger = logging.getLogger(__name__)

# Entry point group name
ENTRY_POINT_GROUP = "molexp.nodes"


def load_plugins(registry: NodeRegistry | None = None) -> None:
    """Discover and load all node plugins via entry points.
    
    This function:
    1. Discovers all entry points in the "molexp.nodes" group
    2. Loads each registration function
    3. Calls it with the Node Registry
    4. Handles errors gracefully (logs and continues)
    
    Args:
        registry: Node registry to use (defaults to global registry)
    """
    if registry is None:
        registry = get_node_registry()
    
    # Check if already loaded
    if registry.is_loaded():
        logger.debug("Plugins already loaded, skipping")
        return
    
    logger.info(f"Loading plugins from entry point group: {ENTRY_POINT_GROUP}")
    
    # Discover entry points
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python 3.9 compatibility
        eps = entry_points().get(ENTRY_POINT_GROUP, [])
    
    loaded_count = 0
    error_count = 0
    
    for ep in eps:
        try:
            logger.debug(f"Loading plugin: {ep.name} from {ep.value}")
            
            # Load the registration function
            register_func: Callable[[NodeRegistry], None] = ep.load()
            
            # Call it with the registry
            register_func(registry)
            
            loaded_count += 1
            logger.info(f"Successfully loaded plugin: {ep.name}")
            
        except Exception as e:
            error_count += 1
            logger.error(
                f"Failed to load plugin '{ep.name}' from '{ep.value}': {e}",
                exc_info=True
            )
            # Continue loading other plugins
            continue
    
    # Mark registry as loaded
    registry.mark_loaded()
    
    logger.info(
        f"Plugin loading complete: {loaded_count} loaded, {error_count} failed"
    )
