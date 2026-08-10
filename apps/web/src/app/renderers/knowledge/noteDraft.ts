/**
 * Pure, DOM-free helpers for the Note editor.
 *
 * `index.md` is the single source of truth for a Note's body — the editor
 * serializes to markdown on save and never introduces block-JSON. These
 * functions carry the only automatable behavior of the editor: a stable,
 * idempotent normalization (so WYSIWYG ⇄ source round-trips faithfully) and
 * a dirty-state predicate that ignores cosmetic whitespace differences.
 */

/** Payload the {@link NoteEditor} hands to `workspaceApi.updateNoteDoc`. */
export interface NoteDocUpdate {
  path: string;
  body: string;
}

/**
 * Normalize a markdown body to a stable canonical form:
 * CRLF/CR line-endings collapse to LF, per-line trailing whitespace is
 * trimmed, and trailing blank lines are dropped. Idempotent by
 * construction — `normalizeMarkdown(normalizeMarkdown(x)) === normalizeMarkdown(x)` —
 * and stable on the empty body (`normalizeMarkdown("") === ""`).
 */
export function normalizeMarkdown(text: string): string {
  return text
    .replace(/\r\n?/g, "\n") // CRLF / lone CR → LF
    .split("\n")
    .map((line) => line.replace(/[ \t]+$/, "")) // trim per-line trailing whitespace
    .join("\n")
    .replace(/\n+$/, ""); // drop trailing blank lines
}

/**
 * Whether `current` differs from `original` after normalization — a
 * trailing-whitespace-only or line-ending-only edit is NOT dirty.
 */
export function isDirty(original: string, current: string): boolean {
  return normalizeMarkdown(original) !== normalizeMarkdown(current);
}

/** Build the normalized update payload sent to the workspace facade. */
export function buildNoteDocUpdate(path: string, body: string): NoteDocUpdate {
  return { path, body: normalizeMarkdown(body) };
}
