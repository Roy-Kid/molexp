"""LogAsset — append-only text stream.

Lives at ``run_dir/logs/<name>.log``.  Supports tail and iterator streaming
for SSE.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Iterator, Literal

from .base import Asset


class LogAsset(Asset):
    """Append-only text log."""

    kind: Literal["log"] = "log"
    encoding: str = "utf-8"
    line_count: int = 0

    def append(self, scope_dir: Path, line: str) -> None:
        """Append a line; the trailing newline is added if missing."""
        target = self.absolute_path(scope_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = line if line.endswith("\n") else line + "\n"
        with open(target, "a", encoding=self.encoding) as fh:
            fh.write(payload)

    def tail(self, scope_dir: Path, n: int = 100) -> list[str]:
        """Return the last ``n`` lines (without trailing newline)."""
        target = self.absolute_path(scope_dir)
        if not target.exists():
            return []
        with open(target, encoding=self.encoding) as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=n)]

    def stream(self, scope_dir: Path) -> Iterator[str]:
        """Yield every existing line.

        Caller is responsible for polling if live tailing is needed;
        this method reads the file in its current state and returns.
        """
        target = self.absolute_path(scope_dir)
        if not target.exists():
            return
        with open(target, encoding=self.encoding) as fh:
            for line in fh:
                yield line.rstrip("\n")

    def size_bytes(self, scope_dir: Path) -> int:
        target = self.absolute_path(scope_dir)
        return os.path.getsize(target) if target.exists() else 0
