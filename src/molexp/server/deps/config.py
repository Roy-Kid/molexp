"""Server settings loaded via molcfg.

Part of :mod:`molexp.server.deps`; the historical import path
``molexp.server.dependencies`` re-exports this surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from molcfg import Config, ConfigLoader, DictSource, Source
from pydantic import BaseModel

_SERVER_DEFAULTS: dict[str, Any] = {
    "workspace_path": "",
    "debug": False,
}


class Settings(BaseModel):
    """Application settings loaded via molcfg."""

    workspace_path: str = ""
    debug: bool = False

    @classmethod
    def from_config(cls, config: Config | None = None) -> Settings:
        """Create settings from a molcfg Config."""
        if config is None:
            config = _load_server_config()
        return cls(
            workspace_path=str(config.get("workspace_path", "")),
            debug=bool(config.get("debug", False)),
        )

    def get_workspace_path(self) -> Path:
        """Get workspace path, defaulting to current directory."""
        if self.workspace_path:
            return Path(self.workspace_path)
        return Path.cwd()


def _load_server_config() -> Config:
    """Load server configuration from defaults + optional molexp.toml."""
    sources: list[Source] = [DictSource(_SERVER_DEFAULTS)]
    config_file = Path.cwd() / "molexp.toml"
    if config_file.exists():
        from molcfg import TomlFileSource

        sources.append(TomlFileSource(str(config_file)))
    return ConfigLoader(sources).load()


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings.from_config()
