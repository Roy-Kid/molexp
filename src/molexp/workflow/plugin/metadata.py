"""Metadata models for task plugins."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PortMetadata(BaseModel):
    """Metadata for a task input or output port.

    Attributes:
        name: Port identifier
        type: Type descriptor ("string", "number", "boolean", "object", "array", "any")
        description: Human-readable description
        required: Whether this port is required
    """

    name: str = Field(..., description="Port identifier")
    type: str = Field(..., description="Type descriptor")
    description: str = Field(default="", description="Port description")
    required: bool = Field(default=True, description="Whether port is required")


class TaskMetadata(BaseModel):
    """Metadata for a task type.

    Attributes:
        label: Human-readable name
        category: Task category (e.g., "io", "text", "http")
        description: Detailed description
        inputs: Input port definitions
        outputs: Output port definitions
        icon: Optional icon name
        tags: Optional tags for search/filtering
    """

    label: str = Field(..., description="Human-readable name")
    category: str = Field(..., description="Task category")
    description: str = Field(..., description="Detailed description")
    inputs: List[PortMetadata] = Field(default_factory=list, description="Input ports")
    outputs: List[PortMetadata] = Field(
        default_factory=list, description="Output ports"
    )
    icon: Optional[str] = Field(default=None, description="Icon name")
    tags: List[str] = Field(default_factory=list, description="Search tags")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return self.model_dump(mode="json")
