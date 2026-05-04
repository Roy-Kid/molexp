"""Credential availability + connection probe helpers (admin-route surface)."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from molexp.agent.model import ModelConfig
from molexp.plugins.model_pydanticai.provider import build_model
from molexp.plugins.model_pydanticai.store import (
    ENV_VAR_FOR_PROVIDER,
    SUPPORTED_PROVIDERS,
    mask_api_key,
)

PROBE_PROMPT = "Reply with the single word 'pong'."
PROBE_MAX_TOKENS = 16
PROBE_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class CredentialStatus:
    """Where the agent will source its API key — or why it can't.

    ``source`` is one of:
      * ``"stored"`` — workspace ``provider.json`` has a key set
      * ``"env"`` — falling back to the provider's standard env var
      * ``"none"`` — neither is set; ``ready`` is False
    """

    ready: bool
    provider: str
    model: str
    source: str
    reason: str = ""
    env_var: str = ""


def check_credentials(config: ModelConfig) -> CredentialStatus:
    """Decide whether the agent can authenticate without starting it.

    Used by ``GET /agent/health`` and the session-creation pre-flight
    so the UI can fail fast with a "configure the provider" hint.
    """

    env_var = ENV_VAR_FOR_PROVIDER.get(config.provider_name, "")
    if config.api_key:
        return CredentialStatus(
            ready=True,
            provider=config.provider_name,
            model=config.model,
            source="stored",
            env_var=env_var,
        )
    if env_var and os.environ.get(env_var):
        return CredentialStatus(
            ready=True,
            provider=config.provider_name,
            model=config.model,
            source="env",
            env_var=env_var,
        )
    if env_var:
        reason = (
            f"No API key configured for provider '{config.provider_name}'. "
            f"Save one in Agent Settings → Provider, or export {env_var} "
            "in the server env."
        )
    else:
        reason = f"No API key configured for provider '{config.provider_name}'."
    return CredentialStatus(
        ready=False,
        provider=config.provider_name,
        model=config.model,
        source="none",
        reason=reason,
        env_var=env_var,
    )


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a minimal end-to-end test against a provider."""

    ok: bool
    latency_ms: int
    reply: str = ""
    error: str | None = None


async def probe_provider(config: ModelConfig) -> ProbeResult:
    """Send a minimal request to the configured provider and return the result.

    Used by the "Test connection" UI to verify the provider works
    end-to-end before the user starts a real session. Wraps the call
    in a hard timeout; the returned error never echoes the API key.
    """

    if config.provider_name not in SUPPORTED_PROVIDERS:
        return ProbeResult(
            ok=False, latency_ms=0, error=f"Unsupported provider '{config.provider_name}'"
        )
    if not config.api_key:
        return ProbeResult(
            ok=False,
            latency_ms=0,
            error="No API key configured. Save one before testing.",
        )

    try:
        from pydantic_ai import Agent
    except ImportError as exc:
        return ProbeResult(ok=False, latency_ms=0, error=f"pydantic-ai not available: {exc}")

    try:
        model = build_model(config)
    except Exception as exc:  # noqa: BLE001 — surface as probe error
        return ProbeResult(ok=False, latency_ms=0, error=f"Invalid config: {exc}")

    agent: Agent = Agent(model=model)
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            agent.run(
                PROBE_PROMPT,
                model_settings={"max_tokens": PROBE_MAX_TOKENS},
            ),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeResult(
            ok=False,
            latency_ms=elapsed,
            error=f"Timeout after {PROBE_TIMEOUT_SECONDS:.0f}s — check network or base_url.",
        )
    except Exception as exc:  # noqa: BLE001 — surface as probe error
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeResult(
            ok=False,
            latency_ms=elapsed,
            error=_format_probe_error(exc, config.api_key or ""),
        )

    elapsed = int((time.monotonic() - started) * 1000)
    reply = str(getattr(result, "output", "")).strip()
    return ProbeResult(ok=True, latency_ms=elapsed, reply=reply)


def _format_probe_error(exc: Exception, api_key: str) -> str:
    raw = str(exc) or exc.__class__.__name__
    if api_key and api_key in raw:
        raw = raw.replace(api_key, mask_api_key(api_key))
    if len(raw) > 400:
        raw = raw[:400] + "…"
    return f"{exc.__class__.__name__}: {raw}"


@dataclass(frozen=True)
class ProviderConfigPublic:
    """Safe public view of :class:`ModelConfig` — never includes the raw key."""

    provider_name: str
    model: str
    base_url: str = ""
    api_key_preview: str = ""
    api_key_set: bool = False
    instructions: str = ""


def to_public(config: ModelConfig) -> ProviderConfigPublic:
    return ProviderConfigPublic(
        provider_name=config.provider_name,
        model=config.model,
        base_url=config.base_url or "",
        api_key_preview=mask_api_key(config.api_key),
        api_key_set=bool(config.api_key),
        instructions=config.instructions,
    )


__all__ = [
    "CredentialStatus",
    "PROBE_MAX_TOKENS",
    "PROBE_PROMPT",
    "PROBE_TIMEOUT_SECONDS",
    "ProbeResult",
    "ProviderConfigPublic",
    "check_credentials",
    "probe_provider",
    "to_public",
]
