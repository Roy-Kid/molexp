"""Debug and inspection nodes."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..node import Node

logger = logging.getLogger(__name__)


class InspectDataConfig(BaseModel):
    """Configuration for InspectDataNode.

    Attributes:
        label: Label for log message
        log_level: Logging level ("debug", "info", "warning", "error")
    """

    label: str = Field(default="Data", description="Label for log message")
    log_level: str = Field(default="info", description="Logging level")


class InspectDataNode(Node[InspectDataConfig, Any]):
    """Inspect and log data for debugging.

    This node logs the input data and passes it through unchanged.
    Useful for debugging workflows.
    Configuration (label, log_level) must be provided at construction.
    """

    config_type = InspectDataConfig

    def execute(self, data: Any) -> Any:
        """Log data and pass through using self.config.

        Args:
            data: Data to inspect

        Returns:
            Same data (pass-through)
        """
        log_func = getattr(logger, self.config.log_level.lower(), logger.info)
        log_func(f"{self.config.label}: {data}")

        return data
