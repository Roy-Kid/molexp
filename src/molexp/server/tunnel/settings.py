"""Resolve how to punch a hole: CLI flags overlay ``molexp config`` ``tunnel.*``.

Never reads the environment. Tokens and binary paths belong in
``~/.molexp/config.json`` or on the ``molexp serve`` command line.

::

    molexp config set tunnel.via zrok
    molexp config set tunnel.token <token>
    molexp config set tunnel.bin /usr/local/bin/zrok
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molexp.server.tunnel.base import TunnelError

VIA_CLOUDFLARED = "cloudflared"
VIA_ZROK = "zrok"
KNOWN_VIA = (VIA_CLOUDFLARED, VIA_ZROK)

_DEFAULT_MODE = {
    VIA_CLOUDFLARED: "quick",
    VIA_ZROK: "public",
}


@dataclass(frozen=True, slots=True)
class TunnelSettings:
    """Provider-agnostic hole-punch settings (local port is not stored here)."""

    via: str
    mode: str
    token: str | None = None
    bin: str | None = None
    hostname: str | None = None


def _as_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_tunnel_settings(
    *,
    via: str | None = None,
    mode: str | None = None,
    token: str | None = None,
    bin: str | None = None,
    hostname: str | None = None,
    config: dict[str, Any] | None = None,
) -> TunnelSettings:
    """Merge CLI overrides over the operator-config ``tunnel`` section.

    *config* is the already-loaded operator file (tests inject a dict).
    ``None`` loads ``~/.molexp/config.json`` via the shared loader.
    """
    if config is None:
        from molexp.services.operator_config import load_operator_config

        config = load_operator_config()
    section = config.get("tunnel")
    stored = section if isinstance(section, dict) else {}

    chosen_via = (via or _as_str(stored.get("via")) or VIA_CLOUDFLARED).lower()
    if chosen_via not in KNOWN_VIA:
        known = ", ".join(KNOWN_VIA)
        raise TunnelError(f"unknown --via {chosen_via!r}; use {known}")

    chosen_mode = (mode or _as_str(stored.get("mode")) or _DEFAULT_MODE[chosen_via]).lower()
    return TunnelSettings(
        via=chosen_via,
        mode=chosen_mode,
        token=token or _as_str(stored.get("token")),
        bin=bin or _as_str(stored.get("bin")),
        hostname=hostname or _as_str(stored.get("hostname")),
    )
