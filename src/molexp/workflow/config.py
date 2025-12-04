"""Configuration utilities for nodes."""

from pydantic import BaseModel


class EmptyConfig(BaseModel):
    """Empty configuration for nodes that don't require parameters.
    
    This is used for nodes that are fully determined by their inputs
    and don't need additional configuration.
    """
    
    model_config = {"frozen": True}
