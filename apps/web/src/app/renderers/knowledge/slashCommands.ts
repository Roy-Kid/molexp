/**
 * Slash-command → markdown mapping (pure; no React, no I/O).
 *
 * The Notion-style "/" menu inserts plain markdown into the Milkdown editor —
 * `index.md` stays the single source of truth, so a command is nothing more
 * than the markdown snippet it drops at the cursor. This module is the one
 * binding, unit-tested unit of the knowledge-docs-07 leaf: every advertised
 * command id maps to a markdown snippet, and an unknown id yields a safe empty
 * default (never throws) so a stale/unknown menu id can never corrupt a doc.
 */

/** Metadata for one entry in the "/" block-insert menu. */
export interface SlashCommandDef {
  /** Stable id — the key into the markdown snippet map. */
  id: string;
  /** Human label shown in the command list. */
  label: string;
  /** One-line description shown beside the label. */
  description: string;
  /** Extra fuzzy-search terms (cmdk matches label + keywords). */
  keywords: string[];
}

/**
 * Canonical id → markdown snippet map. Each snippet is a leading fragment (or a
 * full block skeleton) inserted at the cursor; all are valid CommonMark / GFM so
 * the round-trip through Milkdown never loses structure.
 */
const SNIPPETS: Record<string, string> = {
  heading1: "# ",
  heading2: "## ",
  heading3: "### ",
  bulletList: "- ",
  orderedList: "1. ",
  checkbox: "- [ ] ",
  quote: "> ",
  divider: "---\n",
  codeBlock: "```\n\n```\n",
  table: "| Column 1 | Column 2 |\n| --- | --- |\n|  |  |\n",
};

/** The ordered set of block-insert commands surfaced in the "/" menu. */
export const SLASH_COMMANDS: SlashCommandDef[] = [
  {
    id: "heading1",
    label: "Heading 1",
    description: "Large section heading",
    keywords: ["h1", "title", "#"],
  },
  {
    id: "heading2",
    label: "Heading 2",
    description: "Medium section heading",
    keywords: ["h2", "subtitle", "##"],
  },
  {
    id: "heading3",
    label: "Heading 3",
    description: "Small section heading",
    keywords: ["h3", "###"],
  },
  {
    id: "bulletList",
    label: "Bulleted list",
    description: "A simple bullet point",
    keywords: ["ul", "unordered", "dash"],
  },
  {
    id: "orderedList",
    label: "Numbered list",
    description: "A numbered list item",
    keywords: ["ol", "ordered", "1."],
  },
  {
    id: "checkbox",
    label: "To-do",
    description: "A task-list checkbox",
    keywords: ["todo", "task", "check"],
  },
  { id: "quote", label: "Quote", description: "A blockquote", keywords: ["blockquote", "cite"] },
  {
    id: "divider",
    label: "Divider",
    description: "A horizontal rule",
    keywords: ["hr", "rule", "separator"],
  },
  {
    id: "codeBlock",
    label: "Code block",
    description: "A fenced code block",
    keywords: ["code", "fence", "pre"],
  },
  { id: "table", label: "Table", description: "A 2×2 GFM table", keywords: ["grid", "gfm"] },
];

/**
 * Resolve a slash-command id to the markdown snippet it inserts.
 *
 * Returns `""` for any unknown id — the safe default that keeps an unrecognized
 * menu selection a no-op rather than a thrown error or a corrupted document.
 */
export const slashCommandMarkdown = (id: string): string => SNIPPETS[id] ?? "";
