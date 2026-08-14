"""Tunnel backends punch a hole to a local HTTP port — nothing else.

Provider choice, tokens, and binary paths come from the CLI or
``molexp config`` (``tunnel.*``). Backends never read the environment.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable


class TunnelError(RuntimeError):
    """Tunnel binary missing, failed to start, or misconfigured."""


def http_origin(host: str, port: int) -> str:
    """``http://host:port``, with brackets around a raw IPv6 address."""
    h = host.strip()
    if h.startswith("[") and h.endswith("]"):
        return f"http://{h}:{port}"
    if ":" in h:
        return f"http://[{h}]:{port}"
    return f"http://{h}:{port}"


@runtime_checkable
class TunnelBackend(Protocol):
    """Minimal lifecycle for a reverse tunnel to a local HTTP port."""

    @property
    def public_url(self) -> str | None:
        """HTTPS URL once known, else None."""
        ...

    @property
    def binary(self) -> str:
        """Resolved client binary path."""
        ...

    def start(self) -> None:
        """Spawn the tunnel process (non-blocking)."""
        ...

    def stop(self, *, timeout: float = 5.0) -> None:
        """Terminate the tunnel process group."""
        ...

    def wait_for_url(self, *, timeout: float = 30.0) -> str | None:
        """Block until a public URL is known, or *timeout* elapses."""
        ...


def locate_tunnel_bin(
    names: tuple[str, ...],
    *,
    explicit: str | Path | None = None,
    cache_as: str | None = None,
) -> str | None:
    """Find an existing client. Never downloads.

    Order: *explicit* → ``PATH`` → ``~/.local/bin/<name>``.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        return None
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    from molexp.server.tunnel.fetch import bin_dir

    seen: set[str] = set()
    for name in (cache_as, *names):
        if not name or name in seen:
            continue
        seen.add(name)
        cached = bin_dir() / name
        if cached.is_file() and os.access(cached, os.X_OK):
            return str(cached.resolve())
    return None


def resolve_tunnel_bin(
    names: tuple[str, ...],
    *,
    explicit: str | Path | None = None,
    hint: str,
    cache_as: str | None = None,
) -> str:
    """Return an existing tunnel client, or raise :class:`TunnelError`.

    Does not download. Call :func:`molexp.server.tunnel.fetch.ensure_tunnel_client`
    from the CLI when the operator may be asked first.
    """
    found = locate_tunnel_bin(names, explicit=explicit, cache_as=cache_as)
    if found is not None:
        return found
    if explicit is not None:
        raise TunnelError(f"tunnel binary not executable: {Path(explicit).expanduser()}")
    from molexp.server.tunnel.fetch import bin_dir

    cache_name = cache_as or names[0]
    listed = " or ".join(names)
    raise TunnelError(
        f"--tunnel needs {listed} (not on PATH, not in {bin_dir() / cache_name}).\n"
        f"Re-run in a TTY to download, or: molexp config set tunnel.bin /path/to/{cache_name}\n"
        f"{hint}"
    )


class ChildProcessTunnel:
    """Spawn one tunnel client, scrape a public URL from its logs, kill it."""

    def __init__(
        self,
        *,
        cmd: list[str],
        binary: str,
        parse_line: Callable[[str], str | None],
        label: str,
        on_url: Callable[[str], None] | None = None,
        log_lines: list[str] | None = None,
        initial_url: str | None = None,
    ) -> None:
        self._cmd = cmd
        self._binary = binary
        self._parse_line = parse_line
        self._label = label
        self._on_url = on_url
        self._log_lines = log_lines if log_lines is not None else []
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._public_url = initial_url

    @property
    def public_url(self) -> str | None:
        return self._public_url

    @property
    def binary(self) -> str:
        return self._binary

    def start(self) -> None:
        """Spawn the client (non-blocking)."""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name=f"{self._label}-log",
            daemon=True,
        )
        self._reader.start()

    def wait_for_url(self, *, timeout: float = 30.0) -> str | None:
        """Block until a public URL is parsed or *timeout* elapses."""
        if self._public_url:
            return self._public_url
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._public_url:
                return self._public_url
            if self._proc is not None and self._proc.poll() is not None:
                raise TunnelError(
                    f"{self._label} exited before publishing a URL\n"
                    + "\n".join(self._log_lines[-20:])
                )
            time.sleep(0.1)
        return self._public_url

    def stop(self, *, timeout: float = 5.0) -> None:
        """Terminate the tunnel process group."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            return
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            self._proc = None
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                if sys.platform == "win32":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)
        self._proc = None

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                text = line.rstrip("\n")
                self._log_lines.append(text)
                if self._public_url is None:
                    url = self._parse_line(text)
                    if url:
                        self._public_url = url
                        if self._on_url is not None:
                            self._on_url(url)
        except Exception:
            pass
