"""LLM provider configuration: workspace-scoped storage for API keys & model.

Persists to ``<workspace_root>/.agent_provider.json`` via atomic temp+rename
so concurrent route handlers cannot interleave updates. The file holds the
plaintext API key — never return it through the HTTP API; expose only the
masked preview and an ``is_set`` flag (see :func:`mask_api_key`).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

PROVIDER_FILE = ".agent_provider.json"

PROBE_PROMPT = "Reply with the single word 'pong'."
PROBE_MAX_TOKENS = 16
PROBE_TIMEOUT_SECONDS = 15.0

# Env vars pydantic-ai consults when the corresponding provider has no
# explicit api_key. Used by :func:`check_credentials` to give the user a
# precise hint about what's missing.
ENV_VAR_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

ProviderName = Literal[
    "anthropic", "openai", "google", "deepseek", "openai-compatible"
]

SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = (
    "anthropic",
    "openai",
    "google",
    "deepseek",
    "openai-compatible",
)

DEFAULT_MODELS: dict[ProviderName, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
    "openai-compatible": "gpt-4o",
}

# DeepSeek exposes an OpenAI-compatible API at this endpoint. We
# substitute it automatically when the user picks "deepseek" but leaves
# ``base_url`` empty, so the common case is a single click + paste of
# the API key.
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class ProviderConfig(BaseModel):
    """Active LLM provider configuration for an agent runtime."""

    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""
    base_url: str = ""
    # Workspace-default system prompt addendum, appended to the molexp built-in
    # system prompt on every agent session. Empty means "use the built-in default
    # only". Can be further extended per skill or replaced per session — see
    # ``_pydantic_ai/system_prompt.compose_system_prompt``.
    instructions: str = ""

    def pydantic_ai_model_string(self) -> str:
        """Return the ``"<provider>:<model>"`` string consumed by pydantic-ai."""
        # ``openai-compatible`` reuses the OpenAI implementation but with a
        # custom base_url; the model string still uses the ``openai:`` prefix.
        prefix = "openai" if self.provider == "openai-compatible" else self.provider
        return f"{prefix}:{self.model}"


class ProviderConfigPublic(BaseModel):
    """Safe public view of :class:`ProviderConfig` — never includes the raw key."""

    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str = ""
    api_key_preview: str = ""
    api_key_set: bool = False
    instructions: str = ""


def mask_api_key(key: str) -> str:
    """Mask an API key for display: ``"sk-...1234"`` (last 4 chars).

    Empty input returns an empty string. Keys shorter than 8 chars are
    fully masked to avoid leaking short-key entropy.
    """
    if not key:
        return ""
    if len(key) < 8:
        return "*" * len(key)
    return f"{key[:3]}...{key[-4:]}"


def to_public(config: ProviderConfig) -> ProviderConfigPublic:
    """Project a :class:`ProviderConfig` to its public, key-redacted form."""
    return ProviderConfigPublic(
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        api_key_preview=mask_api_key(config.api_key),
        api_key_set=bool(config.api_key),
        instructions=config.instructions,
    )


class ProviderStore:
    """File-backed store for the workspace's :class:`ProviderConfig`.

    Reads parse the JSON file on demand; writes are atomic and serialized
    by an internal lock. Missing or unparseable files yield the default
    config so the UI can render an empty state without erroring.
    """

    def __init__(self, root: str | Path) -> None:
        self._path = Path(root) / PROVIDER_FILE
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> ProviderConfig:
        if not self._path.exists():
            return ProviderConfig()
        try:
            content = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return ProviderConfig()
        if not isinstance(content, dict):
            return ProviderConfig()
        try:
            return ProviderConfig.model_validate(content)
        except Exception:
            return ProviderConfig()

    def save(self, config: ProviderConfig) -> ProviderConfig:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))
            os.replace(tmp, self._path)
        return config

    def update(
        self,
        *,
        provider: ProviderName | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str | None = None,
    ) -> ProviderConfig:
        """Patch fields. ``api_key=""`` clears the key; ``None`` leaves it.

        Switching ``provider`` without a new ``model`` resets the model to the
        provider's default — different providers use different model names so
        carrying over a stale name would silently break the next session.

        ``instructions`` follows the same convention as ``api_key``: an empty
        string clears the workspace-default prompt; ``None`` leaves it
        unchanged.
        """
        current = self.load()
        next_provider = provider if provider is not None else current.provider
        next_model = model
        if next_model is None:
            if provider is not None and provider != current.provider:
                next_model = DEFAULT_MODELS[provider]
            else:
                next_model = current.model
        next_api_key = api_key if api_key is not None else current.api_key
        next_base_url = base_url if base_url is not None else current.base_url
        next_instructions = (
            instructions if instructions is not None else current.instructions
        )
        updated = ProviderConfig(
            provider=next_provider,
            model=next_model,
            api_key=next_api_key,
            base_url=next_base_url,
            instructions=next_instructions,
        )
        return self.save(updated)


# ── Probe (test connection) ────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a minimal end-to-end test against a provider."""

    ok: bool
    latency_ms: int
    reply: str = ""
    error: str | None = None


async def probe_provider(config: ProviderConfig) -> ProbeResult:
    """Send a minimal request to the configured provider and return the result.

    Used by the "Test connection" UI to verify that ``provider``, ``model``,
    ``api_key``, and ``base_url`` actually work end-to-end before the user
    starts a real session. Wraps the call in a hard timeout so a wrong
    ``base_url`` cannot hang the request handler indefinitely. The returned
    error string is intentionally short and never echoes the API key.
    """
    if not config.api_key:
        return ProbeResult(
            ok=False,
            latency_ms=0,
            error="No API key configured. Save one before testing.",
        )

    # Lazy imports — keeps the provider module importable in environments
    # where pydantic-ai's optional model dependencies are missing.
    try:
        from pydantic_ai import Agent

        from ._pydantic_ai.runtime import _build_model_from_config
    except ImportError as exc:
        return ProbeResult(ok=False, latency_ms=0, error=f"pydantic-ai not available: {exc}")

    try:
        model = _build_model_from_config(config)
    except Exception as exc:
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
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeResult(
            ok=False,
            latency_ms=elapsed,
            error=_format_probe_error(exc, config.api_key),
        )

    elapsed = int((time.monotonic() - started) * 1000)
    reply = str(getattr(result, "output", "")).strip()
    return ProbeResult(ok=True, latency_ms=elapsed, reply=reply)


# ── Credential availability check ─────────────────────────────────────────


@dataclass(frozen=True)
class CredentialStatus:
    """Where the agent will get its API key — or why it can't.

    ``source`` is one of:
      * ``"stored"`` — workspace ``.agent_provider.json`` has a key set
      * ``"env"`` — falling back to the provider's standard env var
      * ``"none"`` — neither is set; ``ready`` will be False
    """

    ready: bool
    provider: str
    model: str
    source: str
    reason: str = ""
    env_var: str = ""


def check_credentials(config: ProviderConfig) -> CredentialStatus:
    """Decide whether the agent can authenticate without fully starting it.

    Used by both the health endpoint (``GET /api/agent/health``) and the
    session-creation pre-flight so the UI can fail fast with a clear
    "configure the provider" hint instead of letting the agent kick off
    and then crash asynchronously inside the LLM client.
    """
    env_var = ENV_VAR_FOR_PROVIDER.get(config.provider, "")
    if config.api_key:
        return CredentialStatus(
            ready=True,
            provider=config.provider,
            model=config.model,
            source="stored",
            env_var=env_var,
        )
    if env_var and os.environ.get(env_var):
        return CredentialStatus(
            ready=True,
            provider=config.provider,
            model=config.model,
            source="env",
            env_var=env_var,
        )
    reason = (
        f"No API key configured for provider '{config.provider}'. "
        f"Save one in Agent Settings → Provider, or export {env_var} in the server env."
        if env_var
        else f"No API key configured for provider '{config.provider}'."
    )
    return CredentialStatus(
        ready=False,
        provider=config.provider,
        model=config.model,
        source="none",
        reason=reason,
        env_var=env_var,
    )


def _format_probe_error(exc: Exception, api_key: str) -> str:
    """Render a probe error for the UI: short, key-redacted, type-prefixed."""
    raw = str(exc) or exc.__class__.__name__
    # Defensive: if the SDK ever embeds the key in an error message, scrub it.
    if api_key and api_key in raw:
        raw = raw.replace(api_key, mask_api_key(api_key))
    # Cap length so a verbose stack trace doesn't blow the UI cell.
    if len(raw) > 400:
        raw = raw[:400] + "…"
    return f"{exc.__class__.__name__}: {raw}"
