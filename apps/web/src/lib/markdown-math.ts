/**
 * Math normalization for the markdown → remark-math → KaTeX pipeline.
 *
 * remark-math only understands `$…$` (inline) and `$$…$$` (display). LLMs
 * routinely emit LaTeX delimiters `\(...\)` / `\[...\]` instead. Bare
 * backslash-paren is *not* special to CommonMark, so `\(` becomes `(` and the
 * formula renders as raw text like `(\nu) in (R_g \sim N^\nu)` — the bug
 * operators report as "公式没渲染".
 *
 * Pipeline:
 *   1. `normalizeLatexDelimiters` — `\(...\)` → `$…$`, `\[...\]` → fenced `$$`
 *   2. `normalizeDisplayMath` — whole-line `$$…$$` → three-line display fence
 *      (remark-math only treats display as *flow* when fences are alone on lines)
 *
 * Both steps skip fenced code (``` / ~~~).
 */

const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
const WHOLE_LINE_DISPLAY_RE = /^(\s{0,3})\$\$(.+)\$\$\s*$/;

/** Split text into alternating outside/code segments (code keeps fences). */
const splitCodeFences = (text: string): { code: boolean; text: string }[] => {
  const parts: { code: boolean; text: string }[] = [];
  let fenceChar: string | null = null;
  let buf: string[] = [];
  const flush = (code: boolean): void => {
    if (buf.length === 0) return;
    parts.push({ code, text: buf.join("\n") });
    buf = [];
  };
  for (const line of text.split("\n")) {
    const fence = line.match(FENCE_RE);
    if (fence) {
      if (fenceChar === null) {
        flush(false);
        fenceChar = fence[1][0];
        buf.push(line);
      } else if (fence[1][0] === fenceChar) {
        buf.push(line);
        flush(true);
        fenceChar = null;
      } else {
        buf.push(line);
      }
      continue;
    }
    buf.push(line);
  }
  flush(fenceChar !== null);
  return parts;
};

/**
 * Convert LaTeX `\(...\)` / `\[...\]` to remark-math `$` / `$$` form.
 * Non-greedy; does not cross already-dollar math (no nested `$` inside).
 */
export const normalizeLatexDelimiters = (text: string): string => {
  if (!text.includes("\\(") && !text.includes("\\[")) return text;
  return splitCodeFences(text)
    .map((part) => {
      if (part.code) return part.text;
      // Display first so `\[` is not partially eaten by a later rule.
      let out = part.text.replace(/\\\[((?:\\.|[^\\])+?)\\\]/g, (_m, body: string) => {
        const inner = body.trim();
        return inner ? `$$\n${inner}\n$$` : _m;
      });
      out = out.replace(/\\\(((?:\\.|[^\\])+?)\\\)/g, (_m, body: string) => {
        const inner = body.trim();
        // Keep single-line inline; multi-line → display block.
        if (!inner) return _m;
        if (inner.includes("\n")) return `$$\n${inner}\n$$`;
        return `$${inner}$`;
      });
      return out;
    })
    .join("\n");
};

/** Rewrite whole-line `$$…$$` into fenced display-math blocks. */
export const normalizeDisplayMath = (text: string): string => {
  if (!text.includes("$$")) return text;
  let fenceChar: string | null = null;
  const lines = text.split("\n").map((line) => {
    const fence = line.match(FENCE_RE);
    if (fence) {
      if (fenceChar === null) fenceChar = fence[1][0];
      else if (fence[1][0] === fenceChar) fenceChar = null;
      return line;
    }
    if (fenceChar !== null) return line;
    const match = line.match(WHOLE_LINE_DISPLAY_RE);
    if (!match) return line;
    const [, indent, body] = match;
    if (body.includes("$$") || !body.trim()) return line;
    return `${indent}$$\n${indent}${body.trim()}\n${indent}$$`;
  });
  return lines.join("\n");
};

/** Full prep before remark-math: LaTeX delimiters + display-fence shape. */
export const prepareMarkdownMath = (text: string): string =>
  normalizeDisplayMath(normalizeLatexDelimiters(text));
