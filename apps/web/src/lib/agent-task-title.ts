/**
 * Display-layer title derivation for agent tasks.
 *
 * A task's `goal` is the user's free-form (often markdown) draft, and the
 * server's auto-derived `title` is merely the whitespace-compacted goal —
 * so both can leak raw markdown syntax (`# Heading`, `**bold**`, list
 * bullets) into list rows, breadcrumbs and page headers. Plan tasks,
 * however, persist a curated report title worth preferring.
 *
 * `agentTaskDisplayTitle` picks the curated title when one exists,
 * otherwise falls back to the first meaningful line of the goal draft,
 * strips markdown markers, and clamps the result.
 */

const UNTITLED = "Untitled agent task";
const DEFAULT_MAX_LENGTH = 72;

/** Strip leading block markers and inline markdown from a single line. */
export const stripMarkdownLine = (line: string): string => {
  const text = line
    .trim()
    // Block prefixes: #/## headings, > blockquotes, -/*/+ bullets, 1. / 1) lists.
    .replace(/^(?:#{1,6}|>+|[-*+]|\d+[.)])\s+/u, "")
    // Links and images: keep the label, drop the URL.
    .replace(/!?\[([^\]]*)\]\([^)]*\)/gu, "$1")
    // Emphasis / inline-code wrappers.
    .replace(/(\*\*|__|[*_`~])/gu, "");
  return text.replace(/\s+/gu, " ").trim();
};

const compact = (text: string): string => text.split(/\s+/u).filter(Boolean).join(" ");

/**
 * True when `title` is just the server's auto-derivation of `goal`
 * (whitespace-compacted, possibly truncated with a "..." suffix) rather
 * than a curated title such as a plan's report title.
 */
export const isAutoDerivedTitle = (title: string, goal: string): boolean => {
  const stem = (title.endsWith("...") ? title.slice(0, -3) : title).trimEnd();
  if (!stem) return true;
  return compact(goal).startsWith(stem);
};

export interface TitledAgentTask {
  title?: string | null;
  goal?: string | null;
}

/**
 * Human-facing task title: curated `title` when available, else the first
 * non-empty line of the goal — markdown-stripped and clamped to
 * `maxLength` characters (ellipsis appended when clipped).
 */
export const agentTaskDisplayTitle = (
  task: TitledAgentTask,
  maxLength: number = DEFAULT_MAX_LENGTH,
): string => {
  const goal = task.goal ?? "";
  const title = (task.title ?? "").trim();
  const curated = title && !isAutoDerivedTitle(title, goal) ? title : "";
  const source = curated || goal;
  const firstLine =
    source
      .split("\n")
      .map(stripMarkdownLine)
      .find((line) => line.length > 0) ?? "";
  if (!firstLine) return UNTITLED;
  if (firstLine.length <= maxLength) return firstLine;
  return `${firstLine.slice(0, Math.max(1, maxLength - 1)).trimEnd()}…`;
};
