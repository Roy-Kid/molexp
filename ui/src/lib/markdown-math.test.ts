/**
 * `normalizeDisplayMath` contract (see markdown-math.ts): whole-line `$$…$$`
 * must reach remark-math in the fenced three-line form (the only shape it
 * parses as display math), while inline math, fenced code, and ambiguous
 * lines pass through untouched.
 */

import { describe, expect, it } from "@rstest/core";

import { normalizeDisplayMath } from "./markdown-math";

describe("normalizeDisplayMath", () => {
  it("rewrites a whole-line $$…$$ into a fenced display block", () => {
    const src = "The pair potential is\n\n$$V(r) = 4\\varepsilon(\\sigma/r)^{12}$$\n\ntail";
    expect(normalizeDisplayMath(src)).toBe(
      "The pair potential is\n\n$$\nV(r) = 4\\varepsilon(\\sigma/r)^{12}\n$$\n\ntail",
    );
  });

  it("keeps mid-sentence $$…$$ inline", () => {
    const src = "depth $$E_{min} = -\\varepsilon$$ at the well";
    expect(normalizeDisplayMath(src)).toBe(src);
  });

  it("keeps single-dollar inline math untouched", () => {
    const src = "minimum at $r_{min} = 2^{1/6}\\sigma$ with depth $-\\varepsilon$";
    expect(normalizeDisplayMath(src)).toBe(src);
  });

  it("leaves already-fenced display math alone", () => {
    const src = "$$\nE = mc^2\n$$";
    expect(normalizeDisplayMath(src)).toBe(src);
  });

  it("never rewrites inside fenced code blocks", () => {
    const src = '```python\nprice = "$$100$$"\n```\n\n$$E = mc^2$$';
    expect(normalizeDisplayMath(src)).toBe('```python\nprice = "$$100$$"\n```\n\n$$\nE = mc^2\n$$');
  });

  it("handles tilde fences too", () => {
    const src = "~~~\n$$raw$$\n~~~";
    expect(normalizeDisplayMath(src)).toBe(src);
  });

  it("skips lines carrying two formulas (would merge into one bogus block)", () => {
    const src = "$$a$$ and $$b$$";
    expect(normalizeDisplayMath(src)).toBe(src);
  });

  it("skips empty and bare-fence lines", () => {
    expect(normalizeDisplayMath("$$$$")).toBe("$$$$");
    expect(normalizeDisplayMath("$$ $$")).toBe("$$ $$");
    expect(normalizeDisplayMath("no math here")).toBe("no math here");
  });

  it("preserves indentation of the rewritten block", () => {
    expect(normalizeDisplayMath("  $$x$$")).toBe("  $$\n  x\n  $$");
  });
});
