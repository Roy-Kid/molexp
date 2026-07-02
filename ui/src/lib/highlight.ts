/**
 * Minimal syntax tokenizer for the read-only code deliverables (python +
 * yaml) shown in the agent Deliverables panel.
 *
 * Deliberately NOT CodeMirror: the repo's @codemirror/* packages are
 * undeclared transitive dependencies of @milkdown/crepe (they may vanish on a
 * Milkdown upgrade and `package.json` is frozen), and a read-only highlight
 * needs none of the editor machinery — @codemirror/view alone is an order of
 * magnitude larger than this file. A handful of token classes (comments,
 * strings, keywords, numbers, decorators, yaml keys) is what review
 * readability actually needs.
 *
 * Contract: `highlightCode` is a pure function and the concatenation of the
 * returned token texts is always exactly the input (nothing dropped, nothing
 * reordered) — the renderer can map tokens 1:1 to styled spans.
 */

export type TokenKind =
  | "plain"
  | "comment"
  | "string"
  | "keyword"
  | "number"
  | "decorator"
  | "key";

export interface HighlightToken {
  text: string;
  kind: TokenKind;
}

export type HighlightLanguage = "python" | "yaml";

const PYTHON_KEYWORDS = new Set([
  "False",
  "None",
  "True",
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "case",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "match",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
]);

// Ordered alternation: triple-quoted strings > single-line strings > comments
// > decorators > numbers > identifiers (classified as keyword or plain).
const PYTHON_TOKEN_RE = new RegExp(
  [
    String.raw`[rRbBuUfF]{0,2}"""[\s\S]*?(?:"""|$)`,
    String.raw`[rRbBuUfF]{0,2}'''[\s\S]*?(?:'''|$)`,
    String.raw`[rRbBuUfF]{0,2}"(?:\\.|[^"\\\n])*"`,
    String.raw`[rRbBuUfF]{0,2}'(?:\\.|[^'\\\n])*'`,
    String.raw`#[^\n]*`,
    String.raw`@[A-Za-z_][\w.]*`,
    String.raw`\b(?:0[xXoObB][\da-fA-F_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[jJ]?)\b`,
    String.raw`\b[A-Za-z_]\w*\b`,
  ].join("|"),
  "g",
);

const classifyPython = (text: string): TokenKind => {
  const head = text[0];
  if (head === "#") return "comment";
  if (head === "@") return "decorator";
  if (/^[rRbBuUfF]{0,2}["']/.test(text)) return "string";
  if (/^\d/.test(text)) return "number";
  return PYTHON_KEYWORDS.has(text) ? "keyword" : "plain";
};

const tokenizePython = (code: string): HighlightToken[] => {
  const tokens: HighlightToken[] = [];
  let cursor = 0;
  PYTHON_TOKEN_RE.lastIndex = 0;
  for (const match of code.matchAll(PYTHON_TOKEN_RE)) {
    const start = match.index;
    if (start > cursor) tokens.push({ text: code.slice(cursor, start), kind: "plain" });
    tokens.push({ text: match[0], kind: classifyPython(match[0]) });
    cursor = start + match[0].length;
  }
  if (cursor < code.length) tokens.push({ text: code.slice(cursor), kind: "plain" });
  return tokens;
};

// yaml is line-oriented: a leading `key:` per line, then strings, comments
// (a `#` opens a comment only at line start or after whitespace), scalars.
const YAML_KEY_RE = /^(\s*(?:-\s+)?)([^\s#'"{[][^:#\n]*):(?=\s|$)/;
const YAML_REST_RE = new RegExp(
  [
    String.raw`"(?:\\.|[^"\\])*"`,
    String.raw`'(?:''|[^'])*'`,
    String.raw`(?:^|(?<=\s))#[^\n]*`,
    String.raw`\b(?:true|false|null|True|False|Null|~)\b`,
    String.raw`(?<![\w.-])[+-]?\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.-])`,
  ].join("|"),
  "g",
);

const classifyYaml = (text: string): TokenKind => {
  const head = text[0];
  if (head === "#") return "comment";
  if (head === '"' || head === "'") return "string";
  if (/^(?:true|false|null|True|False|Null|~)$/.test(text)) return "keyword";
  return "number";
};

const tokenizeYamlLine = (line: string): HighlightToken[] => {
  const tokens: HighlightToken[] = [];
  let rest = line;
  const keyMatch = line.match(YAML_KEY_RE);
  if (keyMatch) {
    const [, lead, key] = keyMatch;
    if (lead) tokens.push({ text: lead, kind: "plain" });
    tokens.push({ text: key, kind: "key" });
    tokens.push({ text: ":", kind: "plain" });
    rest = line.slice(keyMatch[0].length);
  }
  let cursor = 0;
  YAML_REST_RE.lastIndex = 0;
  for (const match of rest.matchAll(YAML_REST_RE)) {
    const start = match.index;
    if (start > cursor) tokens.push({ text: rest.slice(cursor, start), kind: "plain" });
    tokens.push({ text: match[0], kind: classifyYaml(match[0]) });
    cursor = start + match[0].length;
  }
  if (cursor < rest.length) tokens.push({ text: rest.slice(cursor), kind: "plain" });
  return tokens;
};

const tokenizeYaml = (code: string): HighlightToken[] => {
  const tokens: HighlightToken[] = [];
  const lines = code.split("\n");
  lines.forEach((line, i) => {
    tokens.push(...tokenizeYamlLine(line));
    if (i < lines.length - 1) tokens.push({ text: "\n", kind: "plain" });
  });
  return tokens;
};

/** Merge adjacent plain tokens so the renderer emits fewer spans. */
const mergePlain = (tokens: HighlightToken[]): HighlightToken[] => {
  const merged: HighlightToken[] = [];
  for (const token of tokens) {
    if (token.text === "") continue;
    const last = merged[merged.length - 1];
    if (last && last.kind === "plain" && token.kind === "plain") {
      merged[merged.length - 1] = { text: last.text + token.text, kind: "plain" };
    } else {
      merged.push(token);
    }
  }
  return merged;
};

/**
 * Tokenize `code` for display. Unknown/absent languages return the whole
 * text as one plain token, so callers can render every deliverable through
 * the same path.
 */
export const highlightCode = (code: string, language?: string): HighlightToken[] => {
  switch (language) {
    case "python":
      return mergePlain(tokenizePython(code));
    case "yaml":
      return mergePlain(tokenizeYaml(code));
    default:
      return code === "" ? [] : [{ text: code, kind: "plain" }];
  }
};
