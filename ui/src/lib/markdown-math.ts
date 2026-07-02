/**
 * Display-math normalization for the markdown → remark-math pipeline.
 *
 * remark-math (v6 / micromark-extension-math) only recognizes *flow* (display)
 * math when the `$$` fences sit on their own lines. A formula written as a
 * single line — `$$V(r) = 4\varepsilon[...]$$` — parses as `inlineMath`
 * (`math-inline`), so KaTeX renders it in textstyle: fractions squashed, no
 * centering. LLM-generated reports and knowledge notes overwhelmingly use the
 * single-line form to mean display math (verified against the live pipeline:
 * a lone `$$…$$` paragraph yields `.katex`, never `.katex-display`).
 *
 * `normalizeDisplayMath` rewrites a whole-line `$$…$$` into the fenced
 * three-line form remark-math parses as a `math` (display) node. It is
 * deliberately conservative:
 *   - only whole lines (`$$` opens the line, `$$` ends it) are rewritten —
 *     a mid-sentence `$$x$$` stays inline;
 *   - lines inside fenced code blocks (``` / ~~~) are never touched;
 *   - a line carrying more than one `$$…$$` pair is left alone.
 */

const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
const WHOLE_LINE_DISPLAY_RE = /^(\s{0,3})\$\$(.+)\$\$\s*$/;

/** Rewrite whole-line `$$…$$` into fenced display-math blocks. */
export const normalizeDisplayMath = (text: string): string => {
  if (!text.includes("$$")) return text;
  let fenceChar: string | null = null;
  const lines = text.split("\n").map((line) => {
    const fence = line.match(FENCE_RE);
    if (fence) {
      // Opening/closing fenced code — flip state, never rewrite inside.
      if (fenceChar === null) fenceChar = fence[1][0];
      else if (fence[1][0] === fenceChar) fenceChar = null;
      return line;
    }
    if (fenceChar !== null) return line;
    const match = line.match(WHOLE_LINE_DISPLAY_RE);
    if (!match) return line;
    const [, indent, body] = match;
    // Two formulas on one line ("$$a$$ … $$b$$") — greedy capture would merge
    // them into one bogus block; leave the line for inline parsing instead.
    if (body.includes("$$") || !body.trim()) return line;
    return `${indent}$$\n${indent}${body.trim()}\n${indent}$$`;
  });
  return lines.join("\n");
};
