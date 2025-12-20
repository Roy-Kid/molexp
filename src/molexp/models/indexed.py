"""Base models for indexed folder entities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexedEntityBase(BaseModel):
    """Base model for all indexed folder entities.

    This provides common fields and behavior for all molexp-managed
    entities (projects, experiments, runs, assets). Each entity type
    should either inherit from this or provide compatible properties.
    """

    # Core identification
    id: str = Field(..., description="Stable internal identifier")
    kind: Literal["project", "experiment", "run", "asset"] = Field(
        ..., description="Entity type/kind"
    )

    # Human-readable metadata
    name: str = Field(..., description="Human-friendly name")
    description: str = Field(default="", description="Entity description")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )

    # Versioning and extensibility
    schema_version: str = Field(
        default="1.0", description="Schema version for migrations"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extensible metadata storage"
    )

    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"
