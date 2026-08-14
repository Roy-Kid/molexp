"""Cloudflare Tunnel client — punch a hole with ``cloudflared``.

Client resolution: ``--tunnel-bin`` / ``tunnel.bin`` → PATH →
``~/.local/bin/cloudflared``. A missing client is not fetched here — the
CLI asks first. Tokens come from ``--tunnel-token`` or ``molexp config
set tunnel.token`` — never from the environment.

Quick mode::

    cloudflared tunnel --url http://127.0.0.1:{port} --no-autoupdate

Named mode (user-provisioned tunnel)::

    cloudflared tunnel run --token <token>
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from molexp.server.tunnel.base import (
    ChildProcessTunnel,
    TunnelError,
    http_origin,
    resolve_tunnel_bin,
)

# trycloudflare.com quick tunnels and generic https URLs cloudflared prints.
_QUICK_URL_RE = re.compile(
    r"https://[a-z0-9-]+\.trycloudflare\.com",
    re.IGNORECASE,
)
_HTTPS_URL_RE = re.compile(r"https://[^\s\"']+", re.IGNORECASE)

_INSTALL_HINT = (
    "Official builds: https://github.com/cloudflare/cloudflared/releases\n"
    "  Or: molexp config set tunnel.bin /path/to/cloudflared"
)


def resolve_cloudflared_bin(explicit: str | Path | None = None) -> str:
    """Return path to an existing cloudflared or raise :class:`TunnelError`."""
    return resolve_tunnel_bin(
        ("cloudflared",),
        explicit=explicit,
        hint=_INSTALL_HINT,
        cache_as="cloudflared",
    )


def parse_quick_tunnel_url(line: str) -> str | None:
    """Extract a trycloudflare.com URL from one log line, if present."""
    m = _QUICK_URL_RE.search(line)
    return m.group(0) if m else None


def _parse_any_https(line: str) -> str | None:
    m = _HTTPS_URL_RE.search(line)
    if not m:
        return None
    url = m.group(0).rstrip(".,);]")
    # Ignore Cloudflare API endpoints that appear in verbose logs.
    if "trycloudflare.com" in url or "cfargotunnel.com" in url:
        return url
    return None


def _parse_cloudflared_line(line: str) -> str | None:
    url = parse_quick_tunnel_url(line)
    if url is not None:
        return url
    candidate = _parse_any_https(line)
    if candidate and "trycloudflare.com" in candidate:
        return candidate
    return None


class CloudflaredTunnel(ChildProcessTunnel):
    """Manage a cloudflared child process for one local HTTP port."""

    def __init__(
        self,
        *,
        local_port: int,
        mode: str = "quick",
        hostname: str | None = None,
        token: str | None = None,
        binary: str | Path | None = None,
        local_host: str = "127.0.0.1",
        on_url: Callable[[str], None] | None = None,
        log_lines: list[str] | None = None,
    ) -> None:
        if mode not in ("quick", "named"):
            raise TunnelError(f"unknown cloudflared mode {mode!r}; use quick or named")
        resolved = resolve_cloudflared_bin(binary)
        token = token.strip() if isinstance(token, str) and token.strip() else None
        initial: str | None = None
        if mode == "quick":
            cmd = [
                resolved,
                "tunnel",
                "--url",
                http_origin(local_host, local_port),
                "--no-autoupdate",
            ]
        else:
            if not token:
                raise TunnelError(
                    "named tunnel requires --tunnel-token or `molexp config set tunnel.token`"
                )
            cmd = [resolved, "tunnel", "run", "--token", token]
            if hostname:
                initial = hostname if hostname.startswith("http") else f"https://{hostname}"

        super().__init__(
            cmd=cmd,
            binary=resolved,
            parse_line=_parse_cloudflared_line,
            label="cloudflared",
            on_url=on_url,
            log_lines=log_lines,
            initial_url=initial,
        )
        self._mode = mode
