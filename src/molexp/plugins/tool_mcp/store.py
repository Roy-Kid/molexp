"""Multi-scope MCP server store (User + Workspace) with secret separation.

Mirrors VSCode's settings model: two layers — User (``~/.molexp/mcp.json``)
and Workspace (``<root>/.mcp.json``). When a server name appears in both,
the Workspace entry **fully replaces** the User one (no per-field merge).

Secrets live in dedicated files (``.mcp_secrets.json``) that never appear
in any public API response. Public configs reference them via the
``${SECRET:KEY}`` placeholder syntax. Resolution is strict: only the
Workspace and User secret stores are consulted — there is **no fallback
to ``os.environ``**. A missing secret marks the server entry as having
``unresolved_secrets`` and the runtime skips it.

The on-disk JSON format stays compatible with Claude-Code-style
``.mcp.json`` files, with one addition: each entry **must** carry an
explicit ``"type"`` discriminator (``stdio`` / ``http`` / ``sse`` /
``sse``). Entries missing the discriminator are reported as
invalid rather than silently inferred.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter, ValidationError

from .resources.base import _format_validation_error, _now_iso

# ── Constants ──────────────────────────────────────────────────────────────

MCP_CONFIG_FILENAME = ".mcp.json"
MCP_SECRETS_FILENAME = ".mcp_secrets.json"
USER_DIR = Path.home() / ".molexp"

# ``${SECRET:KEY_NAME}`` — only env/header *values* may carry these.
SECRET_REF_PATTERN = re.compile(r"\$\{SECRET:([A-Za-z_][A-Za-z0-9_]*)\}")

# Server names become JSON keys, file fragments in logs, and tool prefixes
# inside pydantic-ai. Restrict to a path-safe ASCII alphabet.
SERVER_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

HTTP_TRANSPORTS: tuple[str, ...] = ("http", "sse")
ALL_TRANSPORTS: tuple[str, ...] = ("stdio", *HTTP_TRANSPORTS)


# ── Scope enum ─────────────────────────────────────────────────────────────


class McpScope(str, Enum):
    """Two-tier scope mirroring VSCode User vs Workspace settings."""

    USER = "user"
    WORKSPACE = "workspace"


# ── Spec models (discriminated by ``type``) ────────────────────────────────


class StdioSpec(BaseModel):
    """Local subprocess MCP server spec."""

    type: Literal["stdio"]
    command: str = Field(min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class OAuth2AuthSpec(BaseModel):
    """OAuth 2.0 Authorization Code + PKCE auth for a remote MCP server.

    Only the discovery URL is required — the rest (client registration,
    PKCE challenge, token exchange, refresh) is handled by the MCP SDK's
    ``OAuthClientProvider`` against ``HttpSpec.url``. ``scopes`` defaults to
    empty (server picks); ``client_id`` is optional and only needed when the
    target authorization server doesn't support Dynamic Client Registration.
    """

    type: Literal["oauth2"]
    scopes: list[str] = Field(default_factory=list)
    client_id: str | None = Field(default=None, max_length=512)


class HttpSpec(BaseModel):
    """Remote HTTP MCP server spec.

    Two transports are accepted:

    - ``http`` — modern streamable HTTP wire format, matching Claude Code's
      ``.mcp.json`` convention. Use this for any new server.
    - ``sse`` — legacy long-poll transport, kept only for older servers
      that haven't migrated to streamable HTTP yet.

    The historical ``streamable-http`` value is normalized to ``http`` on
    read (see :func:`_read_servers`); it is not accepted on write.
    """

    type: Literal["http", "sse"]
    url: str = Field(min_length=1, max_length=4096)
    headers: dict[str, str] = Field(default_factory=dict)
    # Optional structured auth. ``None`` means "use the static headers as-is"
    # (Bearer / Basic / API Key flows already encode their token there via
    # ``${SECRET:KEY}``). When set, the runtime injects the corresponding
    # auth provider and ignores any ``Authorization`` in ``headers``. Today
    # only OAuth 2.0 is supported; once a second auth type lands, re-wrap
    # this field as ``Annotated[Union[...], Discriminator("type")]``.
    auth: OAuth2AuthSpec | None = None


McpServerSpec = Annotated[Union[StdioSpec, HttpSpec], Discriminator("type")]
_SPEC_ADAPTER: TypeAdapter[StdioSpec | HttpSpec] = TypeAdapter(McpServerSpec)


# ── Public entry view ──────────────────────────────────────────────────────


class AuthSummary(BaseModel):
    """Public-facing description of an HTTP server's auth configuration.

    Only fields safe to surface in the UI: the auth type and (for OAuth)
    the requested scopes. Token values, client secrets, and refresh tokens
    never appear here — they live in the OAuth token store.
    """

    model_config = ConfigDict(frozen=True)

    type: str  # "oauth2" — extend with new variants here
    scopes: tuple[str, ...] = ()
    client_id: str | None = None


class McpServerEntry(BaseModel):
    """One MCP server, possibly merged across scopes.

    Carries enough metadata for the UI to render the row + alert on
    missing secrets, but never the secret values themselves. Fields are
    sourced from the on-disk ``.mcp.json`` plus computed read-time
    annotations (``shadowed``/``valid``/``unresolved_secrets``).

    Pydantic ``BaseModel`` (replaces the historical
    ``@dataclass(frozen=True)``): ``model_config`` declares
    ``frozen=True`` so the entry is immutable after construction —
    callers that need a copy with adjusted ``shadowed`` flag must use
    :py:meth:`pydantic.BaseModel.model_copy(update=…)`. Every field is
    typed; legacy callers that built entries with positional args can
    instead pass keyword arguments unchanged.

    ``created_at`` / ``updated_at`` are persisted alongside the entry
    so the settings UI can show "added on" + "last edited" times. They
    default to empty strings to keep migration of older config files
    painless — the next write fills them.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    scope: McpScope
    transport: str
    command: str | None
    args: tuple[str, ...]
    url: str | None
    env_keys: tuple[str, ...]
    header_keys: tuple[str, ...]
    secret_refs: tuple[str, ...]
    unresolved_secrets: tuple[str, ...]
    shadowed: bool = False
    valid: bool = True
    invalid_reason: str = ""
    auth: AuthSummary | None = None
    created_at: str = ""
    updated_at: str = ""


class ResolvedSpec(BaseModel):
    """Spec with secrets substituted in env/headers, ready for the runtime.

    The runtime is the only legitimate consumer; never serialize this.
    """

    model_config = ConfigDict(frozen=True)

    transport: str
    command: str | None
    args: tuple[str, ...]
    url: str | None
    env: dict[str, str]
    headers: dict[str, str]
    auth: AuthSummary | None = None


class UnresolvedSecretError(Exception):
    """Raised when a server references a secret missing from both stores."""

    def __init__(self, server: str, keys: list[str]) -> None:
        self.server = server
        self.keys = keys
        super().__init__(f"Server '{server}' references unresolved secrets: {', '.join(keys)}")


# ── Secrets store (single-file KV) ─────────────────────────────────────────


class McpSecretsStore:
    """File-backed KV for MCP-related secrets.

    Format: ``{"secrets": {"KEY": "value", ...}}``. Atomic writes via
    temp-file + ``os.replace``; the temp file is chmod'd ``0600`` before
    the rename so the secret is never readable by other users on POSIX.
    Best-effort on Windows (chmod is a no-op).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def get(self, key: str) -> str | None:
        return self._load_raw().get(key)

    def list_keys(self) -> list[str]:
        return sorted(self._load_raw().keys())

    def set(self, key: str, value: str) -> None:
        """Write a key. Empty value deletes the key (delete-by-clear UX)."""
        with self._lock:
            current = self._load_raw()
            if value == "":
                current.pop(key, None)
            else:
                current[key] = value
            self._write(current)

    def delete(self, key: str) -> bool:
        with self._lock:
            current = self._load_raw()
            if key not in current:
                return False
            current.pop(key)
            self._write(current)
            return True

    def _load_raw(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            content = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        secrets = content.get("secrets") if isinstance(content, dict) else None
        if not isinstance(secrets, dict):
            return {}
        return {str(k): str(v) for k, v in secrets.items() if isinstance(v, str)}

    def _write(self, secrets: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"secrets": secrets}, indent=2, ensure_ascii=False))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)


# ── MCP store (the aggregate) ──────────────────────────────────────────────


class McpStore:
    """Two-tier MCP config store: User + Workspace, VSCode-style.

    Read paths merge both layers with Workspace taking precedence on
    name collision. Mutations always target an explicit scope (caller
    chooses where to write). Secret resolution consults Workspace
    secrets first, then User secrets — strict, no env-var fallback.

    Mirrors :class:`SkillStore` / :class:`ToolStore` for parity:
    accepts an optional ``user_home_dir`` to override the default
    ``~/.molexp/`` location (testing convenience). All public mutators
    serialize through ``_lock`` so concurrent route handlers cannot
    interleave reads/writes against the same file.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        user_home_dir: str | Path | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._workspace_path = self._workspace_root / MCP_CONFIG_FILENAME
        if user_home_dir is None:
            user_home_dir = USER_DIR
        self._user_dir = Path(user_home_dir)
        self._user_path = self._user_dir / MCP_CONFIG_FILENAME
        self._workspace_secrets = McpSecretsStore(self._workspace_root / MCP_SECRETS_FILENAME)
        self._user_secrets = McpSecretsStore(self._user_dir / MCP_SECRETS_FILENAME)
        self._lock = Lock()

    # ── Path / store accessors ─────────────────────────────────────────────

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def config_path(self, scope: McpScope) -> Path:
        return self._workspace_path if scope is McpScope.WORKSPACE else self._user_path

    def secrets(self, scope: McpScope) -> McpSecretsStore:
        return self._workspace_secrets if scope is McpScope.WORKSPACE else self._user_secrets

    # ── Read ───────────────────────────────────────────────────────────────

    def list(self) -> list[McpServerEntry]:
        """Return merged view of both scopes, with shadowing info.

        User entries shadowed by a Workspace entry of the same name are
        still emitted (so the UI can show "this exists at User scope but
        is being overridden") with ``shadowed=True``.
        """
        user_specs = _read_servers(self._user_path)
        ws_specs = _read_servers(self._workspace_path)

        entries: list[McpServerEntry] = []
        for name, raw in user_specs.items():
            shadowed = name in ws_specs
            entries.append(self._build_entry(name, raw, McpScope.USER, shadowed))
        for name, raw in ws_specs.items():
            entries.append(self._build_entry(name, raw, McpScope.WORKSPACE, shadowed=False))
        return entries

    def get(self, scope: McpScope, name: str) -> McpServerEntry | None:
        servers = _read_servers(self.config_path(scope))
        if name not in servers:
            return None
        return self._build_entry(name, servers[name], scope, shadowed=False)

    def secret_references(self, scope: McpScope) -> dict[str, list[str]]:
        """Map of secret-key → list of server names that reference it.

        Computed only over the requested scope's config so the UI can
        show "Workspace entries that need GITHUB_TOKEN" without leaking
        which User-scope entries also reference it.
        """
        servers = _read_servers(self.config_path(scope))
        out: dict[str, set[str]] = {}
        for name, raw in servers.items():
            if not isinstance(raw, dict):
                continue
            values: Iterable[Any] = []
            if isinstance(raw.get("env"), dict):
                values = list(values) + list(raw["env"].values())
            if isinstance(raw.get("headers"), dict):
                values = list(values) + list(raw["headers"].values())
            for key in _collect_refs(values):
                out.setdefault(key, set()).add(name)
        return {k: sorted(v) for k, v in out.items()}

    # ── Mutate ─────────────────────────────────────────────────────────────

    def upsert(self, scope: McpScope, name: str, spec: dict[str, Any]) -> McpServerEntry:
        """Create or replace a server entry at the given scope.

        Validates name + spec before touching disk. Empty/invalid input
        raises ``ValueError`` (caller maps to HTTP 400).
        """
        if not SERVER_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid server name '{name}': use lowercase letters, "
                "digits, underscore, hyphen; max 64 chars; must start with "
                "a letter or digit."
            )
        try:
            validated = _SPEC_ADAPTER.validate_python(spec)
        except ValidationError as exc:
            raise ValueError(f"Invalid spec: {_format_validation_error(exc)}") from exc

        if isinstance(validated, HttpSpec) and not _is_http_url(validated.url):
            raise ValueError("URL scheme must be http or https for http/sse transports")

        with self._lock:
            path = self.config_path(scope)
            servers = _read_servers(path)
            now = _now_iso()
            payload = validated.model_dump()
            existing = servers.get(name)
            if isinstance(existing, dict):
                payload["created_at"] = str(existing.get("created_at") or now)
            else:
                payload["created_at"] = now
            payload["updated_at"] = now
            servers[name] = payload
            _write_servers(path, servers)
        return self._build_entry(name, servers[name], scope, shadowed=False)

    def delete(self, scope: McpScope, name: str) -> bool:
        with self._lock:
            path = self.config_path(scope)
            servers = _read_servers(path)
            if name not in servers:
                return False
            servers.pop(name)
            _write_servers(path, servers)
            return True

    # ── Secret substitution (runtime only) ─────────────────────────────────

    def resolve(self, entry: McpServerEntry) -> ResolvedSpec:
        """Substitute ``${SECRET:K}`` in the entry's env/headers.

        Raises :class:`UnresolvedSecretError` if any referenced secret is
        missing. The runtime guards on ``entry.unresolved_secrets`` first
        and skips such entries, so this only raises on race-condition
        edge cases (a secret deleted between list() and resolve()).
        """
        if entry.unresolved_secrets:
            raise UnresolvedSecretError(entry.name, list(entry.unresolved_secrets))

        path = self.config_path(entry.scope)
        servers = _read_servers(path)
        raw = servers.get(entry.name)
        if not isinstance(raw, dict):
            raise KeyError(f"Server '{entry.name}' missing at scope {entry.scope.value}")

        if entry.transport == "stdio":
            env_resolved = {
                k: self._substitute(entry.name, v) for k, v in (raw.get("env") or {}).items()
            }
            return ResolvedSpec(
                transport="stdio",
                command=str(raw.get("command") or ""),
                args=tuple(str(a) for a in (raw.get("args") or [])),
                url=None,
                env=env_resolved,
                headers={},
            )

        header_resolved = {
            k: self._substitute(entry.name, v) for k, v in (raw.get("headers") or {}).items()
        }
        return ResolvedSpec(
            transport=entry.transport,
            command=None,
            args=(),
            url=str(raw.get("url") or ""),
            env={},
            headers=header_resolved,
            auth=entry.auth,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_entry(
        self,
        name: str,
        raw: Any,
        scope: McpScope,
        shadowed: bool,
    ) -> McpServerEntry:
        created_at = ""
        updated_at = ""
        if isinstance(raw, dict):
            created_at = str(raw.get("created_at", "") or "")
            updated_at = str(raw.get("updated_at", "") or "")

        if not isinstance(raw, dict):
            return McpServerEntry(
                name=name,
                scope=scope,
                transport="",
                command=None,
                args=(),
                url=None,
                env_keys=(),
                header_keys=(),
                secret_refs=(),
                unresolved_secrets=(),
                shadowed=shadowed,
                valid=False,
                invalid_reason="entry is not an object",
                created_at=created_at,
                updated_at=updated_at,
            )
        try:
            spec = _SPEC_ADAPTER.validate_python(raw)
        except ValidationError as exc:
            env_keys = (
                tuple(sorted((raw.get("env") or {}).keys()))
                if isinstance(raw.get("env"), dict)
                else ()
            )
            header_keys = (
                tuple(sorted((raw.get("headers") or {}).keys()))
                if isinstance(raw.get("headers"), dict)
                else ()
            )
            return McpServerEntry(
                name=name,
                scope=scope,
                transport=str(raw.get("type") or ""),
                command=raw.get("command") if isinstance(raw.get("command"), str) else None,
                args=tuple(str(a) for a in (raw.get("args") or []))
                if isinstance(raw.get("args"), list)
                else (),
                url=raw.get("url") if isinstance(raw.get("url"), str) else None,
                env_keys=env_keys,
                header_keys=header_keys,
                secret_refs=(),
                unresolved_secrets=(),
                shadowed=shadowed,
                valid=False,
                invalid_reason=_format_validation_error(exc),
                created_at=created_at,
                updated_at=updated_at,
            )

        auth_summary: AuthSummary | None = None
        if isinstance(spec, StdioSpec):
            env_keys = tuple(sorted(spec.env.keys()))
            secret_refs = _collect_refs(spec.env.values())
            command = spec.command
            args: tuple[str, ...] = tuple(spec.args)
            url: str | None = None
            header_keys: tuple[str, ...] = ()
        else:
            env_keys = ()
            header_keys = tuple(sorted(spec.headers.keys()))
            secret_refs = _collect_refs(spec.headers.values())
            command = None
            args = ()
            url = spec.url
            if isinstance(spec.auth, OAuth2AuthSpec):
                auth_summary = AuthSummary(
                    type="oauth2",
                    scopes=tuple(spec.auth.scopes),
                    client_id=spec.auth.client_id,
                )

        unresolved = tuple(k for k in secret_refs if not self._has_secret(k))

        return McpServerEntry(
            name=name,
            scope=scope,
            transport=spec.type,
            command=command,
            args=args,
            url=url,
            env_keys=env_keys,
            header_keys=header_keys,
            secret_refs=secret_refs,
            unresolved_secrets=unresolved,
            shadowed=shadowed,
            valid=True,
            invalid_reason="",
            auth=auth_summary,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _has_secret(self, key: str) -> bool:
        return (
            self._workspace_secrets.get(key) is not None or self._user_secrets.get(key) is not None
        )

    def _substitute(self, server: str, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        missing: list[str] = []

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            v = self._workspace_secrets.get(key)
            if v is None:
                v = self._user_secrets.get(key)
            if v is None:
                missing.append(key)
                return ""
            return v

        result = SECRET_REF_PATTERN.sub(repl, value)
        if missing:
            raise UnresolvedSecretError(server, missing)
        return result


# ── Module-level helpers ───────────────────────────────────────────────────


def _read_servers(path: Path) -> dict[str, Any]:
    """Return the ``mcpServers`` map from ``path``, or empty dict on error.

    Performs a one-shot normalization for legacy values: ``streamable-http``
    in older configs is rewritten to ``http`` so the new schema (which only
    accepts ``http`` / ``sse`` / ``stdio``) can validate. Read-only — the
    mutation only affects the in-memory dict; the file on disk stays as-is
    until a subsequent :func:`_write_servers` call.
    """
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = content.get("mcpServers") if isinstance(content, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for name, spec in raw.items():
        if isinstance(spec, dict) and spec.get("type") == "streamable-http":
            spec = {**spec, "type": "http"}
        out[name] = spec
    return out


def _write_servers(path: Path, servers: dict[str, Any]) -> None:
    """Atomically write ``{mcpServers: ...}`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _collect_refs(values: Iterable[Any]) -> tuple[str, ...]:
    found: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        for m in SECRET_REF_PATTERN.finditer(v):
            found.add(m.group(1))
    return tuple(sorted(found))


def _is_http_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))
