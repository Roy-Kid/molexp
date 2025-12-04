"""Debug and inspection nodes."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..primitives import TransformNode

logger = logging.getLogger(__name__)


class InspectDataConfig(BaseModel):
    """Configuration for InspectDataNode.
    
    Attributes:
        label: Label for log message
        log_level: Logging level ("debug", "info", "warning", "error")
    """
    
    label: str = Field(default="Data", description="Label for log message")
    log_level: str = Field(default="info", description="Logging level")


class InspectDataNode(TransformNode[InspectDataConfig, Any, Any]):
    """Inspect and log data for debugging.
    
    This node logs the input data and passes it through unchanged.
    Useful for debugging workflows.
    """
    
    config_type = InspectDataConfig
    
    def transform(self, data: Any, config: InspectDataConfig) -> Any:
        """Log data and pass through.
        
        Args:
            data: Data to inspect
            config: Configuration with label and log level
            
        Returns:
            Same data (pass-through)
        """
        log_func = getattr(logger, config.log_level.lower(), logger.info)
        log_func(f"{config.label}: {data}")
        
        return data
