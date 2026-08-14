"""Fetch official tunnel clients into ``~/.local/bin`` after the operator agrees.

Downloads happen only when the CLI asked and the user said yes. GitHub
Releases only. ``https_proxy`` is left to urllib for the fetch itself.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from molexp.server.tunnel.base import TunnelError

_GITHUB_API = "https://api.github.com/repos/{repo}/releases/latest"
_USER_AGENT = "molexp-tunnel"
_TIMEOUT = 60.0


def bin_dir() -> Path:
    """Default install location for fetched tunnel clients."""
    return Path.home() / ".local" / "bin"


def platform_pair() -> tuple[str, str]:
    """Return ``(os, arch)`` as used in upstream release asset names."""
    system = sys_platform()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        raise TunnelError(
            f"no official tunnel binary for this CPU ({platform.machine()}). "
            "Download one on a login node and: molexp config set tunnel.bin /path"
        )
    os_name = {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(system)
    if os_name is None:
        raise TunnelError(f"no official tunnel binary for {system}")
    return os_name, arch


def sys_platform() -> str:
    """Indirection so tests can pin the platform without rewriting sys."""
    import sys

    return sys.platform


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/octet-stream, application/vnd.github+json",
        },
    )


def download_url(url: str, dest: Path, *, urlopen: Callable = urllib.request.urlopen) -> None:
    """Write *url* to *dest* atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urlopen(_request(url), timeout=_TIMEOUT) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise TunnelError(f"download failed ({url}): {exc}") from exc


def _extract_member(archive: Path, dest: Path, basenames: tuple[str, ...]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".extract")
    name = archive.name.lower()
    member_file = None
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if Path(info.filename).name in basenames and not info.is_dir():
                    member_file = zf.read(info)
                    break
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            for info in tf.getmembers():
                if Path(info.name).name in basenames and info.isfile():
                    extracted = tf.extractfile(info)
                    if extracted is None:
                        continue
                    member_file = extracted.read()
                    break
    else:
        raise TunnelError(f"unsupported archive: {archive.name}")
    if member_file is None:
        raise TunnelError(f"{archive.name} has no {basenames} binary")
    tmp.write_bytes(member_file)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp.replace(dest)


def github_latest_asset(
    repo: str,
    want: Callable[[str], bool],
    *,
    urlopen: Callable = urllib.request.urlopen,
) -> str:
    """Return ``browser_download_url`` of the first latest-release asset *want* accepts."""
    api = _GITHUB_API.format(repo=repo)
    try:
        with urlopen(_request(api), timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise TunnelError(f"cannot list {repo} releases: {exc}") from exc
    for asset in data.get("assets") or []:
        name = asset.get("name") or ""
        url = asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str) and want(name):
            return url
    raise TunnelError(f"no {repo} release asset matches this platform")


def _install_from_url(
    url: str,
    dest: Path,
    *,
    member_names: tuple[str, ...],
    urlopen: Callable = urllib.request.urlopen,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lower = url.lower()
    if lower.endswith((".tar.gz", ".tgz", ".zip")):
        suffix = ".zip" if lower.endswith(".zip") else ".tar.gz"
        blob = dest.with_name(dest.name + ".dl" + suffix)
        try:
            download_url(url, blob, urlopen=urlopen)
            _extract_member(blob, dest, member_names)
        finally:
            blob.unlink(missing_ok=True)
    else:
        download_url(url, dest, urlopen=urlopen)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def fetch_cloudflared(dest: Path, *, urlopen: Callable = urllib.request.urlopen) -> None:
    """Install the official cloudflared binary at *dest*."""
    os_name, arch = platform_pair()

    def want(name: str) -> bool:
        if os_name == "linux":
            return name == f"cloudflared-linux-{arch}"
        if os_name == "darwin":
            return name in (
                f"cloudflared-darwin-{arch}.tgz",
                f"cloudflared-darwin-{arch}",
            )
        if os_name == "windows":
            return name == f"cloudflared-windows-{arch}.exe"
        return False

    url = github_latest_asset("cloudflare/cloudflared", want, urlopen=urlopen)
    _install_from_url(url, dest, member_names=("cloudflared",), urlopen=urlopen)


def fetch_zrok(dest: Path, *, urlopen: Callable = urllib.request.urlopen) -> None:
    """Install the official zrok binary at *dest*."""
    os_name, arch = platform_pair()
    suffix_tar = f"_{os_name}_{arch}.tar.gz"
    suffix_zip = f"_{os_name}_{arch}.zip"

    def want(name: str) -> bool:
        lower = name.lower()
        if not lower.startswith("zrok"):
            return False
        return lower.endswith((suffix_tar, suffix_zip))

    url = github_latest_asset("openziti/zrok", want, urlopen=urlopen)
    _install_from_url(
        url,
        dest,
        member_names=("zrok", "zrok2"),
        urlopen=urlopen,
    )


def interpret_download_reply(
    raw: str,
    *,
    binary: str,
    default_dir: Path,
) -> Path | None:
    """Parse a ``[y/N/path]`` reply into an install location.

    ``y`` / ``yes`` → ``default_dir / binary``.
    empty / ``n`` / ``no`` → decline.
    Anything else is a custom path: a directory (existing, trailing slash, or
    last component not equal to *binary*) gets ``/binary`` appended.
    """
    text = raw.strip()
    if not text or text.lower() in {"n", "no"}:
        return None
    if text.lower() in {"y", "yes"}:
        return default_dir / binary
    path = Path(text).expanduser()
    as_dir = str(text).endswith(("/", "\\")) or path.is_dir() or path.name != binary
    return (path / binary) if as_dir else path


def ensure_tunnel_client(
    *,
    via: str,
    explicit: str | Path | None,
    ask: Callable[[str, Path], Path | None],
) -> str:
    """Locate a client, or download after *ask* returns a destination.

    *ask(binary, default_dir)* is a CLI hook. Return a file path to download
    there, or ``None`` to abort — nothing is fetched until then.
    """
    from molexp.server.tunnel.base import locate_tunnel_bin
    from molexp.server.tunnel.settings import VIA_CLOUDFLARED, VIA_ZROK

    if via == VIA_CLOUDFLARED:
        names: tuple[str, ...] = ("cloudflared",)
        cache_as = "cloudflared"
        fetch = fetch_cloudflared
    elif via == VIA_ZROK:
        names = ("zrok", "zrok2")
        cache_as = "zrok"
        fetch = fetch_zrok
    else:
        raise TunnelError(f"unknown --via {via!r}")

    found = locate_tunnel_bin(names, explicit=explicit, cache_as=cache_as)
    if found is not None:
        return found
    if explicit is not None:
        raise TunnelError(f"tunnel binary not executable: {Path(explicit).expanduser()}")

    dest = ask(cache_as, bin_dir())
    if dest is None:
        raise TunnelError(
            f"declined download of {cache_as}. "
            f"Install it yourself or: molexp config set tunnel.bin /path/to/{cache_as}"
        )
    dest = dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch(dest)
    if dest.is_file() and os.access(dest, os.X_OK):
        return str(dest.resolve())
    raise TunnelError(f"download finished but {dest} is not executable")
