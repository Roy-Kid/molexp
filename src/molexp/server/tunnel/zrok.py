"""zrok client — punch a hole with the official hosted ``zrok.io`` frontend.

Client resolution: ``--tunnel-bin`` / ``tunnel.bin`` → PATH →
``~/.local/bin/zrok``. A missing client is not fetched here — the CLI
asks first. The machine must still be enabled (``zrok enable``); this
module only shares. Tokens for reserved shares come from
``--tunnel-token`` or ``molexp config set tunnel.token``.

Public mode (ephemeral HTTPS)::

    zrok share public http://127.0.0.1:{port} --headless

Reserved mode (pre-created share token)::

    zrok share reserved <token> --headless
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

# Hosted zrok.io frontends and typical self-hosted ``*.zrok.*`` names.
_ZROK_URL_RE = re.compile(
    r"https://[a-z0-9.-]*zrok[a-z0-9.-]*",
    re.IGNORECASE,
)

_INSTALL_HINT = (
    "Official builds: https://github.com/openziti/zrok/releases\n"
    "  After fetch: zrok enable <account-token>   # once per machine\n"
    "  Or: molexp config set tunnel.bin /path/to/zrok"
)


def resolve_zrok_bin(explicit: str | Path | None = None) -> str:
    """Return path to an existing ``zrok`` / ``zrok2`` or raise :class:`TunnelError`."""
    return resolve_tunnel_bin(
        ("zrok", "zrok2"),
        explicit=explicit,
        hint=_INSTALL_HINT,
        cache_as="zrok",
    )


def parse_zrok_share_url(line: str) -> str | None:
    """Extract a public zrok HTTPS URL from one log line, if present."""
    m = _ZROK_URL_RE.search(line)
    if not m:
        return None
    return m.group(0).rstrip(".,);]")


class ZrokTunnel(ChildProcessTunnel):
    """Manage a zrok share process for one local HTTP port."""

    def __init__(
        self,
        *,
        local_port: int,
        mode: str = "public",
        token: str | None = None,
        binary: str | Path | None = None,
        local_host: str = "127.0.0.1",
        on_url: Callable[[str], None] | None = None,
        log_lines: list[str] | None = None,
    ) -> None:
        if mode not in ("public", "reserved"):
            raise TunnelError(f"unknown zrok mode {mode!r}; use public or reserved")
        resolved = resolve_zrok_bin(binary)
        token = token.strip() if isinstance(token, str) and token.strip() else None
        if mode == "public":
            cmd = [
                resolved,
                "share",
                "public",
                http_origin(local_host, local_port),
                "--headless",
            ]
        else:
            if not token:
                raise TunnelError(
                    "reserved zrok share requires --tunnel-token "
                    "or `molexp config set tunnel.token`"
                )
            cmd = [resolved, "share", "reserved", token, "--headless"]

        super().__init__(
            cmd=cmd,
            binary=resolved,
            parse_line=parse_zrok_share_url,
            label="zrok",
            on_url=on_url,
            log_lines=log_lines,
        )
        self._mode = mode
