"""Public hole-punch for ``molexp serve --tunnel`` (cloudflared, zrok).

This package only starts/stops a tunnel client against a local HTTP port.
Provider settings live in the CLI (``--via``, ``--tunnel-token``, …) or
``molexp config`` (``tunnel.*``) — never environment variables.
"""

from __future__ import annotations

from collections.abc import Callable

from molexp.server.tunnel.base import TunnelBackend, TunnelError
from molexp.server.tunnel.cloudflared import (
    CloudflaredTunnel,
    parse_quick_tunnel_url,
    resolve_cloudflared_bin,
)
from molexp.server.tunnel.fetch import ensure_tunnel_client
from molexp.server.tunnel.settings import (
    VIA_CLOUDFLARED,
    VIA_ZROK,
    TunnelSettings,
    resolve_tunnel_settings,
)
from molexp.server.tunnel.zrok import (
    ZrokTunnel,
    parse_zrok_share_url,
    resolve_zrok_bin,
)


def open_tunnel(
    *,
    local_port: int,
    settings: TunnelSettings,
    local_host: str = "127.0.0.1",
    on_url: Callable[[str], None] | None = None,
) -> TunnelBackend:
    """Build the backend named by *settings* — does not start it."""
    if settings.via == VIA_CLOUDFLARED:
        return CloudflaredTunnel(
            local_port=local_port,
            mode=settings.mode,
            hostname=settings.hostname,
            token=settings.token,
            binary=settings.bin,
            local_host=local_host,
            on_url=on_url,
        )
    if settings.via == VIA_ZROK:
        return ZrokTunnel(
            local_port=local_port,
            mode=settings.mode,
            token=settings.token,
            binary=settings.bin,
            local_host=local_host,
            on_url=on_url,
        )
    raise TunnelError(f"unknown --via {settings.via!r}")


__all__ = [
    "VIA_CLOUDFLARED",
    "VIA_ZROK",
    "CloudflaredTunnel",
    "TunnelBackend",
    "TunnelError",
    "TunnelSettings",
    "ZrokTunnel",
    "ensure_tunnel_client",
    "open_tunnel",
    "parse_quick_tunnel_url",
    "parse_zrok_share_url",
    "resolve_cloudflared_bin",
    "resolve_tunnel_settings",
    "resolve_zrok_bin",
]
