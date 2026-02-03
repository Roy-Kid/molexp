"""Unified configuration for molexp."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MolexpConfig(BaseModel):
    """Runtime configuration settings."""

    plugin_fail_fast: bool = Field(
        True, description="Fail-fast on plugin load errors"
    )


_CONFIG = MolexpConfig()


def get_config() -> MolexpConfig:
    """Return the active configuration."""
    return _CONFIG


def set_config(**kwargs) -> MolexpConfig:
    """Update configuration values."""
    global _CONFIG
    _CONFIG = _CONFIG.model_copy(update=kwargs)
    return _CONFIG
