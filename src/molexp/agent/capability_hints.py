"""Capability discovery hints and trusted namespace policy.

This module is intentionally outside PlanMode.  It may know about
project-specific trusted namespaces, aliases, and conservative text
patterns; PlanMode only consumes the structured hints and evidence that
fall out of discovery.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "LEGACY_VALIDATION_NAMESPACES",
    "TRUSTED_NAMESPACES",
    "TRUSTED_NAMESPACE_ALIASES",
    "CapabilityHint",
    "CapabilityHintPolicy",
    "CapabilityTriggerInput",
    "ConstraintViolation",
    "HintStrength",
    "NoHandRolledLAMMPSDataCheck",
    "TrustedNamespacePolicy",
    "validate_hint_constraints",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_ANY_FROZEN = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

HintStrength = Literal["required", "preferred", "hint"]

TRUSTED_NAMESPACES: tuple[str, ...] = (
    "molpy",
    "molpack",
    "molrs",
    "molexp",
    "lammps",
)
"""Default trusted namespaces that may produce discovery hints."""

LEGACY_VALIDATION_NAMESPACES: tuple[str, ...] = (
    "molpy",
    "molexp",
    "molvis",
    "molpack",
    "molnex",
    "molq",
    "mollog",
    "molcfg",
)
"""Compatibility namespace set for legacy evidence validators."""

TRUSTED_NAMESPACE_ALIASES: dict[str, tuple[str, ...]] = {
    "molcrafts toolchain": ("molpy", "molpack", "molexp"),
    "molcrafts": ("molpy", "molpack", "molrs", "molexp"),
}
"""Conservative alias expansion for trusted namespaces."""


class CapabilityHint(BaseModel):
    """Structured hint used to bias capability discovery.

    Hints never route tools and never authorize codegen to invent APIs.
    They only say what discovery should look for, how strict the user's
    wording was, and which validation constraints should be enforced if
    discovery/codegen proceeds.
    """

    model_config = _FROZEN

    namespace: str
    strength: HintStrength = "hint"
    phrase: str = ""
    query_hints: tuple[str, ...] = ()
    reason: str = ""
    constraint_tags: tuple[str, ...] = ()

    @property
    def requires_evidence(self) -> bool:
        return self.strength == "required"

    @property
    def fallback_allowed(self) -> bool:
        return self.strength != "required"


class CapabilityTriggerInput(BaseModel):
    """Context available to hint policies.

    ``raw_user_input`` is deliberately first-class because explicit API
    constraints can be lost when a plan brief abstracts stages into
    natural-language outcomes.
    """

    model_config = _ANY_FROZEN

    raw_user_input: str = ""
    project_digest: Any | None = None
    plan_brief: Any | None = None
    draft_capability_needs: Any | None = None


@runtime_checkable
class CapabilityHintPolicy(Protocol):
    """Extract discovery hints from request and planning context."""

    def extract(self, input: CapabilityTriggerInput) -> list[CapabilityHint]: ...


class TrustedNamespacePolicy:
    """Default conservative trusted-namespace hint policy."""

    def __init__(
        self,
        *,
        namespaces: Iterable[str] = TRUSTED_NAMESPACES,
        aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._namespaces = tuple(dict.fromkeys(ns.lower() for ns in namespaces))
        self._aliases = aliases if aliases is not None else TRUSTED_NAMESPACE_ALIASES

    def extract(self, input: CapabilityTriggerInput) -> list[CapabilityHint]:
        text = _context_text(input)
        if not text:
            return []

        hints: dict[str, CapabilityHint] = {}
        for sentence in _sentences(text):
            lowered = sentence.lower()
            for phrase, namespaces in self._matches(lowered):
                strength = _classify_strength(lowered, phrase)
                if phrase in self._aliases and strength == "hint":
                    # Broad project aliases are too noisy as casual
                    # mentions. Expand them only when the user's wording
                    # carries required/preferred force.
                    continue
                constraint_tags = _constraint_tags(lowered)
                for namespace in namespaces:
                    hint = CapabilityHint(
                        namespace=namespace,
                        strength=strength,
                        phrase=phrase,
                        query_hints=_query_hints(namespace, phrase, sentence),
                        reason=_reason(strength, phrase, sentence),
                        constraint_tags=constraint_tags,
                    )
                    _merge_hint(hints, hint)
        return list(hints.values())

    def _matches(self, sentence: str) -> list[tuple[str, tuple[str, ...]]]:
        matches: list[tuple[str, tuple[str, ...]]] = []
        matched_aliases: list[str] = []
        for alias, namespaces in sorted(
            self._aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if _contains_phrase(sentence, alias):
                if any(alias in existing for existing in matched_aliases):
                    continue
                matched_aliases.append(alias)
                matches.append((alias, namespaces))
        for namespace in self._namespaces:
            if _contains_phrase(sentence, namespace):
                matches.append((namespace, (namespace,)))
        return matches


class ConstraintViolation(BaseModel):
    """A source-level violation emitted by a hint-specific validator."""

    model_config = _FROZEN

    reason: str
    detail: str


class NoHandRolledLAMMPSDataCheck:
    """Detect obvious hand-written LAMMPS data writers.

    This is intentionally lightweight.  It catches the high-risk pattern
    the policy creates when a user forbids hand-rolled data files:
    generated code containing LAMMPS-data section literals alongside
    direct file writes.
    """

    _SECTION_MARKERS = (
        "Masses",
        "Atoms",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Pair Coeffs",
        "Bond Coeffs",
    )

    def validate(self, source: str) -> tuple[ConstraintViolation, ...]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ()

        string_literals = tuple(_iter_string_literals(tree))
        has_section_literal = any(
            _looks_like_lammps_data_section(value) for value in string_literals
        )
        has_file_write = _has_file_write_call(tree)
        if has_section_literal and has_file_write:
            return (
                ConstraintViolation(
                    reason="forbidden_hand_rolled_output",
                    detail=(
                        "source appears to write LAMMPS data sections directly "
                        "despite a discovery hint forbidding hand-rolled output"
                    ),
                ),
            )
        return ()


def validate_hint_constraints(
    source: str,
    hints: Iterable[CapabilityHint],
) -> tuple[ConstraintViolation, ...]:
    """Run validators activated by hint constraint tags."""

    tags = {tag for hint in hints for tag in hint.constraint_tags}
    violations: list[ConstraintViolation] = []
    if "no_hand_rolled_lammps_data" in tags:
        violations.extend(NoHandRolledLAMMPSDataCheck().validate(source))
    return tuple(violations)


def _context_text(input: CapabilityTriggerInput) -> str:
    parts = [
        input.raw_user_input,
        _stringify(input.project_digest),
        _stringify(input.plan_brief),
        _stringify(input.draft_capability_needs),
    ]
    return "\n".join(part for part in parts if part)


def _stringify(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return str(model_dump_json())
        except Exception:
            return str(value)
    return str(value)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n.;!?]+", text) if part.strip()]


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text) is not None


def _classify_strength(sentence: str, phrase: str) -> HintStrength:
    if _has_preferred_cue(sentence):
        return "preferred"
    if _has_required_cue(sentence, phrase):
        return "required"
    return "hint"


def _has_preferred_cue(sentence: str) -> bool:
    preferred_cues = (
        "prefer",
        "if available",
        "wherever possible",
        "where possible",
        "when possible",
        "as much as possible",
    )
    return any(cue in sentence for cue in preferred_cues)


def _has_required_cue(sentence: str, phrase: str) -> bool:
    required_cues = (
        "must use",
        "must be written via",
        "must be generated via",
        "need to explicitly use",
        "needs to explicitly use",
        "explicitly use",
        "do not hand-roll",
        "do not hand roll",
        "don't hand-roll",
        "don't hand roll",
        "must not hand-roll",
        "must not hand roll",
    )
    if any(cue in sentence for cue in required_cues):
        return True
    return re.search(rf"(?<!\w)use\s+{re.escape(phrase)}(?![\w-])", sentence) is not None


def _constraint_tags(sentence: str) -> tuple[str, ...]:
    hand_roll = (
        "hand-roll" in sentence
        or "hand roll" in sentence
        or "hand-rolled" in sentence
        or "hand rolled" in sentence
    )
    data_file = "lammps data" in sentence or "data file" in sentence
    if hand_roll and data_file:
        return ("no_hand_rolled_lammps_data",)
    return ()


def _query_hints(namespace: str, phrase: str, sentence: str) -> tuple[str, ...]:
    hints = [namespace]
    if phrase != namespace:
        hints.append(phrase)
    if sentence:
        hints.append(sentence[:240])
    return tuple(dict.fromkeys(hints))


def _reason(strength: HintStrength, phrase: str, sentence: str) -> str:
    if strength == "required":
        return f"user explicitly required {phrase!r}: {sentence[:160]}"
    if strength == "preferred":
        return f"user preferred {phrase!r}: {sentence[:160]}"
    return f"user mentioned {phrase!r}: {sentence[:160]}"


_STRENGTH_RANK: dict[HintStrength, int] = {"hint": 0, "preferred": 1, "required": 2}


def _merge_hint(hints: dict[str, CapabilityHint], hint: CapabilityHint) -> None:
    existing = hints.get(hint.namespace)
    if existing is None:
        hints[hint.namespace] = hint
        return
    strength = (
        hint.strength
        if _STRENGTH_RANK[hint.strength] > _STRENGTH_RANK[existing.strength]
        else existing.strength
    )
    hints[hint.namespace] = existing.model_copy(
        update={
            "strength": strength,
            "phrase": existing.phrase or hint.phrase,
            "query_hints": tuple(dict.fromkeys((*existing.query_hints, *hint.query_hints))),
            "reason": existing.reason or hint.reason,
            "constraint_tags": tuple(
                dict.fromkeys((*existing.constraint_tags, *hint.constraint_tags))
            ),
        }
    )


def _iter_string_literals(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            parts = [
                value.value
                for value in node.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            if parts:
                yield "".join(parts)


def _looks_like_lammps_data_section(value: str) -> bool:
    markers = sum(1 for marker in NoHandRolledLAMMPSDataCheck._SECTION_MARKERS if marker in value)
    return markers >= 2 or "\nAtoms\n" in value or "\nMasses\n" in value


def _has_file_write_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"write", "write_text", "writelines"}:
            return True
        if isinstance(func, ast.Name) and func.id == "open":
            return True
    return False
