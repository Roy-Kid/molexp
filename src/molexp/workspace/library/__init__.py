"""Library — notes + literature/references per Folder scope.

A scope's ``library/`` directory (owned by the Folder) holds markdown notes —
each a :class:`~molexp.workspace.assets.note.NoteAsset` registered in the
catalog — and a molexp-native :class:`Reference` store (``references.json``).
:class:`Library` is the storage surface; :meth:`Library.build_index` derives
the agent-/UI-readable :class:`LibraryIndex` (``index.json`` + ``INDEX.md``).
"""

from .index import LibraryIndex, NoteEntry
from .library import Library
from .reference import Reference, ReferenceStore
from .zotero import ZoteroImportError, read_zotero_references

__all__ = [
    "Library",
    "LibraryIndex",
    "NoteEntry",
    "Reference",
    "ReferenceStore",
    "ZoteroImportError",
    "read_zotero_references",
]
