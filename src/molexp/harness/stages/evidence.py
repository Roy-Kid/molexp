"""Diagnose plan failures and gather molmcp evidence for repair loops.

Repair is **not** a lottery: after a failed produce/check the harness extracts
structured diagnoses, looks up symbols via molmcp when available, and either:

* enriches the next attempt's feedback with real signatures / usage, or
* hard-stops with a capability gap so the operator updates the plan or the
  molpy/molmcp index — never invents a missing API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.errors import StageExecutionError

SymbolLookup = Callable[[str], dict[str, Any] | None]

__all__ = [
    "CapabilityGap",
    "Diagnosis",
    "collect_evidence_text",
    "diagnose_failure",
    "is_capability_gap",
]

_LOG = get_logger(__name__)

#: Violation / error codes that mean "this capability does not exist — stop".
#: Note: ``unknown_capability`` is intentionally **not** here — the binder
#: inventing a capability_id that is absent from a non-empty catalog is a
#: retryable LLM error (re-bind from the catalog), not a hard toolchain gap.
_GAP_CODES = frozenset(
    {
        "capability_missing",
        "api_symbol_missing",
        "SYMBOL_NOT_FOUND",
        "CAPABILITY_NOT_FOUND",
    }
)
#: Codes that look like gaps in free text but stay retryable when they only
#: mean "LLM picked a bad id from a live catalog".
_RETRYABLE_BIND_CODES = frozenset({"unknown_capability", "capability_call_invalid"})

_ATTR_ERROR_RE = re.compile(
    r"AttributeError:\s*['\"]?([\w.]+)['\"]?\s+object has no attribute\s+['\"](\w+)['\"]",
    re.IGNORECASE,
)
_NAME_ERROR_RE = re.compile(r"NameError:\s*name\s+['\"](\w+)['\"]\s+is not defined")
_MODULE_ERROR_RE = re.compile(r"No module named\s+['\"]([\w.]+)['\"]")
# Qualnames from installed MolCrafts packages (molrs may appear in
# tracebacks as implementation detail; molmcp no longer indexes it as a
# Python API source — live inspect still helps read error types).
_MOLCRAFTS_QUALNAME_RE = re.compile(r"\b((?:molpy|molpack|molrs|molq|molexp|molnex)(?:\.[\w]+)+)\b")
#: Mock/patch targets the test invented — open the real API page for them.
_MOCK_TARGET_RE = re.compile(r"""patch\(\s*['\"]((?:molpy|molpack)[\w.]*)['\"]""")


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    """A missing molcrafts capability or API symbol — not a retryable LLM fluke."""

    code: str
    message: str
    symbol: str | None = None

    def user_message(self) -> str:
        target = f" ({self.symbol})" if self.symbol else ""
        return (
            f"MolpyCapabilityMissing[{self.code}]{target}: {self.message}. "
            "Update the experiment plan to use an available capability, or update "
            "molpy/molmcp (refresh the index) and re-run the plan."
        )


@dataclass(slots=True)
class Diagnosis:
    """Structured read of one failed produce/check attempt."""

    text: str
    codes: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    gaps: list[CapabilityGap] = field(default_factory=list)

    @property
    def is_capability_gap(self) -> bool:
        return bool(self.gaps)


def diagnose_failure(exc: StageExecutionError | BaseException, text: str) -> Diagnosis:
    """Parse a failure message / persisted report into a :class:`Diagnosis`."""
    codes: list[str] = []
    symbols: list[str] = []
    gaps: list[CapabilityGap] = []

    # Structured validation reports (JSON).
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError, TypeError:
        payload = None
    if isinstance(payload, dict):
        for violation in payload.get("violations") or []:
            if not isinstance(violation, dict):
                continue
            code = str(violation.get("code") or "")
            message = str(violation.get("message") or "")
            if code:
                codes.append(code)
            # Hard gaps only — invented ids ("not in the registry") stay
            # retryable so RepairLoop can re-bind from the catalog.
            if code in _GAP_CODES:
                sym = _first_molcrafts_symbol(message)
                gaps.append(
                    CapabilityGap(
                        code=code,
                        message=message or "capability missing from grounded registry",
                        symbol=sym,
                    )
                )
            symbols.extend(_MOLCRAFTS_QUALNAME_RE.findall(message))

    # Traceback / free-text patterns — collect symbols for evidence lookup.
    for match in _ATTR_ERROR_RE.finditer(text):
        obj, attr = match.group(1), match.group(2)
        symbols.append(f"{obj}.{attr}")
    for match in _NAME_ERROR_RE.finditer(text):
        symbols.append(match.group(1))
    for match in _MODULE_ERROR_RE.finditer(text):
        symbols.append(match.group(1))
    symbols.extend(_MOLCRAFTS_QUALNAME_RE.findall(text))
    # Mock targets (tests inventing constructors) — open the real API.
    symbols.extend(_MOCK_TARGET_RE.findall(text))

    # Dedup while preserving order.
    codes = list(dict.fromkeys(codes))
    symbols = list(dict.fromkeys(symbols))
    # Gaps: dedup by (code, symbol, message)
    seen_gaps: set[tuple[str, str | None, str]] = set()
    unique_gaps: list[CapabilityGap] = []
    for gap in gaps:
        key = (gap.code, gap.symbol, gap.message)
        if key in seen_gaps:
            continue
        seen_gaps.add(key)
        unique_gaps.append(gap)

    # Free-text capability gap phrasing from StageExecutionError messages.
    if not unique_gaps and any(c in text for c in _GAP_CODES):
        unique_gaps.append(
            CapabilityGap(
                code="capability_missing",
                message=text[:500],
                symbol=_first_molcrafts_symbol(text),
            )
        )

    _ = exc  # reserved for typed subclass branches
    return Diagnosis(text=text, codes=codes, symbols=symbols, gaps=unique_gaps)


def is_capability_gap(diagnosis: Diagnosis) -> bool:
    return diagnosis.is_capability_gap


def collect_evidence_text(
    diagnosis: Diagnosis,
    *,
    ctx: HarnessRunContext | None = None,
    lookup: SymbolLookup | None = None,
    promote_misses: bool = True,
) -> str:
    """Build an evidence block for the next repair attempt.

    ``lookup`` is an optional callable ``(symbol: str) -> dict | None`` used in
    tests; production passes nothing and uses in-process molmcp ``open`` /
    search when available. Confirmed misses are appended as
    :class:`CapabilityGap` entries on ``diagnosis`` when ``promote_misses``.
    """
    sections: list[str] = ["## Repair diagnosis"]
    if diagnosis.codes:
        sections.append("codes: " + ", ".join(diagnosis.codes))
    if diagnosis.symbols:
        sections.append("symbols: " + ", ".join(diagnosis.symbols))
    if diagnosis.gaps:
        sections.append("capability_gaps:")
        for gap in diagnosis.gaps:
            sections.append(f"- {gap.user_message()}")

    symbols = list(diagnosis.symbols)
    if not symbols and not diagnosis.gaps:
        return "\n".join(sections)

    resolver = lookup if lookup is not None else _default_lookup

    sections.append("## molmcp evidence (open pages — use these signatures; do not invent APIs)")
    for symbol in symbols[:8]:
        try:
            detail = resolver(symbol)
        except Exception as exc:  # never block repair on evidence I/O
            _LOG.debug(f"[evidence] lookup {symbol!r} failed: {exc!r}")
            detail = None
        if detail is None or (isinstance(detail, dict) and detail.get("ok") is False):
            code = "SYMBOL_NOT_FOUND"
            err = "not found"
            if isinstance(detail, dict):
                code = str(detail.get("code") or code)
                err = str(detail.get("error") or err)
            sections.append(f"- {symbol}: MISS ({code}) — {err}")
            root = symbol.split(".", 1)[0]
            if (
                promote_misses
                and _looks_molcrafts(root)
                and code in {"SYMBOL_NOT_FOUND", "CAPABILITY_NOT_FOUND", "api_symbol_missing"}
            ):
                diagnosis.gaps.append(
                    CapabilityGap(
                        code="api_symbol_missing",
                        message=f"molmcp could not resolve {symbol!r}: {err}",
                        symbol=symbol,
                    )
                )
            # Still try live source even when molmcp misses.
            src_block = _live_source_signature(symbol)
            if src_block:
                sections.append(src_block)
            continue
        # Prefer markdown open-page body when present.
        if isinstance(detail, dict) and detail.get("markdown"):
            md = str(detail["markdown"])[:2500]
            sections.append(f"### open `{symbol}`\n\n{md}")
        else:
            sig = ""
            if isinstance(detail, dict):
                data = detail.get("data") if isinstance(detail.get("data"), dict) else None
                usage = data.get("usage") if isinstance(data, dict) else None
                inner = usage or data or detail.get("detail") or detail
                if isinstance(inner, dict):
                    sig = str(
                        inner.get("signature")
                        or inner.get("summary")
                        or inner.get("docstring")
                        or ""
                    )[:800]
            sections.append(f"- {symbol}: {sig or detail!r}"[:900])
        # molmcp is an index, not ground truth — always attach live inspect.
        src_block = _live_source_signature(symbol)
        if src_block:
            sections.append(src_block)

    sections.append(
        "\n**Repair rules:** Prefer **live source signatures** over inventing APIs. "
        "molmcp points direction; when it conflicts with inspect/getsource, "
        "trust the installed package source. Assert on the task's real return "
        "values / files — do not mock constructors the impl never calls."
    )
    _ = ctx
    return "\n".join(sections)


_COLLECTION_CACHE: object | None = None
_COLLECTION_FAILED = False


def _live_collection() -> object | None:
    """Lazy singleton CollectionIndex for in-process evidence lookups."""
    global _COLLECTION_CACHE, _COLLECTION_FAILED
    if _COLLECTION_FAILED:
        return None
    if _COLLECTION_CACHE is not None:
        return _COLLECTION_CACHE
    try:
        from molmcp.config import load_config  # ty: ignore[unresolved-import]
        from molmcp.runtime import build_collection  # ty: ignore[unresolved-import]

        _COLLECTION_CACHE = build_collection(load_config())
        return _COLLECTION_CACHE
    except Exception as exc:
        _LOG.debug(f"[evidence] build_collection failed: {exc!r}")
        _COLLECTION_FAILED = True
        return None


def _live_source_signature(symbol: str) -> str:
    """Import the installed package and read signature + source snippet.

    molmcp cannot cover every edge (dtype rules, private helpers, version
    skew). After molmcp points at a symbol, this is the ground-truth layer.
    """
    if not symbol or not _looks_molcrafts(symbol.split(".", 1)[0]):
        return ""
    try:
        import importlib
        import inspect

        parts = symbol.split(".")
        # Resolve longest importable prefix, then walk attrs.
        obj: object | None = None
        for i in range(len(parts), 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            obj = mod
            for attr in parts[i:]:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                break
        if obj is None:
            return (
                f"### live source `{symbol}`\n\n*(could not import/resolve in harness process)*\n"
            )
        lines: list[str] = [f"### live source `{symbol}`"]
        try:
            if callable(obj):
                lines.append(f"signature: `{inspect.signature(obj)}`")
            else:
                lines.append(f"type: `{type(obj).__name__}`")
        except TypeError, ValueError:
            lines.append(f"type: `{type(obj).__name__}`")
        try:
            src_file = inspect.getsourcefile(obj) or inspect.getfile(obj)  # type: ignore[arg-type]
            if src_file:
                lines.append(f"file: `{src_file}`")
        except TypeError, OSError:
            pass
        try:
            src = inspect.getsource(obj)  # type: ignore[arg-type]
            # Cap: enough for signature + docstring + a few body lines.
            snippet = src if len(src) <= 1800 else src[:1800] + "\n# … truncated …\n"
            lines.append("```python")
            lines.append(snippet.rstrip())
            lines.append("```")
        except TypeError, OSError:
            doc = inspect.getdoc(obj)
            if doc:
                lines.append(doc[:600])
        return "\n".join(lines) + "\n"
    except Exception as exc:  # never block repair
        _LOG.debug(f"[evidence] live source {symbol!r} failed: {exc!r}")
        return ""


def _default_lookup(symbol: str) -> dict[str, Any] | None:
    """In-process molmcp open/search — prefers real open pages over stubs."""
    collection = _live_collection()
    if collection is None:
        return {
            "ok": False,
            "code": "molmcp_unavailable",
            "error": "molmcp collection not available in harness process",
            "ref": symbol,
        }
    try:
        from molmcp.collection.browse import (  # ty: ignore[unresolved-import]
            open_ref,
            search_scoped,
        )
    except Exception as exc:
        return {
            "ok": False,
            "code": "molmcp_unavailable",
            "error": f"browse import failed: {exc}",
            "ref": symbol,
        }

    # Qualname search → open first hit.
    try:
        hits = search_scoped(collection, symbol, limit=8)
        for item in hits.get("results") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("qualname") or "")
            if symbol in title or title.endswith(symbol.split(".")[-1]):
                ref = item.get("ref")
                if ref:
                    page = open_ref(collection, str(ref))
                    if page.get("ok"):
                        return page
        # Fall back: open first hit if any.
        results = hits.get("results") or []
        if results and isinstance(results[0], dict) and results[0].get("ref"):
            page = open_ref(collection, str(results[0]["ref"]))
            if page.get("ok"):
                return page
    except Exception as exc:
        _LOG.debug(f"[evidence] search/open {symbol!r} failed: {exc!r}")
    return {
        "ok": False,
        "code": "SYMBOL_NOT_FOUND",
        "error": f"no open page for {symbol!r}",
        "ref": symbol,
    }


def _looks_molcrafts(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in {"molpy", "molpack", "molrs", "molq", "molexp", "molnex"}


def _first_molcrafts_symbol(text: str) -> str | None:
    match = _MOLCRAFTS_QUALNAME_RE.search(text)
    return match.group(1) if match else None
