"""Remote execution plugin (HPC / Slurm via molq).

This plugin is loaded lazily by the plugin registry.
It requires ``molq`` to be installed.
"""

from __future__ import annotations

from typing import Any


def get_remote_plugin() -> Any:
    """Entry point called by :class:`~molexp.plugins.PluginRegistry`.

    Raises ``ImportError`` if ``molq`` is not installed.
    """
    from molq.transfer import FileTransfer, TransferSpec  # noqa: F401

    # Return a namespace / facade with the public API
    from .spec import EnvironmentSpec, RemoteSpec

    return _RemotePlugin(
        EnvironmentSpec=EnvironmentSpec,
        RemoteSpec=RemoteSpec,
        TransferSpec=TransferSpec,
        FileTransfer=FileTransfer,
    )


class _RemotePlugin:
    """Thin namespace returned by get_remote_plugin()."""

    def __init__(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)
