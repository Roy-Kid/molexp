"""Cross-host POSIX path — pure path arithmetic, no local I/O.

Subclasses :class:`pathlib.PurePosixPath`.  Pair with
:class:`molexp.workspace.fs.FileSystem` for I/O (the FileSystem layer
decides whether the path lands on disk via :mod:`pathlib` or over SSH via
``molq`` Transport).

Why subclass instead of using ``PurePosixPath`` directly:
  1. Single project-internal type identity — ``isinstance(p, molexp.Path)``
     is a meaningful predicate.
  2. Single import surface — ``from molexp import Path`` instead of
     ``from pathlib import PurePosixPath`` scattered across the codebase.
  3. Future ergonomics (e.g., a ``path.local()`` adapter) can land here
     without affecting unrelated :mod:`pathlib` users.

Why **not** :class:`pathlib.Path` (i.e., the concrete ``PosixPath`` /
``WindowsPath``): :class:`pathlib.Path` binds to *this host's* filesystem.
``Path("/scratch/x")`` on a laptop will silently answer ``.exists()``
against the laptop's ``/scratch``, never the cluster.  ``PurePosixPath``
removes I/O methods so there is no path through which "the wrong
filesystem" can be hit by accident.
"""

from __future__ import annotations

from pathlib import PurePosixPath


class Path(PurePosixPath):
    """A POSIX path that may refer to a local *or* remote filesystem.

    Does no I/O itself — pair with :class:`molexp.workspace.fs.FileSystem`
    to read/write/list.  ``__fspath__`` returns the POSIX string, so callers
    can wrap with :class:`pathlib.Path` for genuine local I/O when
    appropriate.
    """

    __slots__ = ()


__all__ = ["Path"]
