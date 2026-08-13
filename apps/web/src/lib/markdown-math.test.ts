/**
 * Math normalization contract (see markdown-math.ts).
 */

import { describe, expect, it } from "@rstest/core";

import {
  normalizeDisplayMath,
  normalizeLatexDelimiters,
  prepareMarkdownMath,
} from "./markdown-math";

describe("normalizeLatexDelimiters", () => {
  it("rewrites \\(…\\) to $…$ so remark-math sees inline math", () => {
    const src = "scaling exponent \\(\\nu\\) in \\(R_g \\sim N^{\\nu}\\).";
    expect(normalizeLatexDelimiters(src)).toBe("scaling exponent $\\nu$ in $R_g \\sim N^{\\nu}$.");
  });

  it("rewrites \\[…\\] to a fenced display block", () => {
    const src = "The law is\n\n\\[R_g \\sim N^{\\nu}\\]\n\ndone.";
    expect(normalizeLatexDelimiters(src)).toBe("The law is\n\n$$\nR_g \\sim N^{\\nu}\n$$\n\ndone.");
  });

  it("never rewrites inside fenced code", () => {
    const src = "```tex\n\\(x\\)\n```\n\n\\(y\\)";
    expect(normalizeLatexDelimiters(src)).toBe("```tex\n\\(x\\)\n```\n\n$y$");
  });

  it("leaves plain text without latex delimiters alone", () => {
    expect(normalizeLatexDelimiters("no math")).toBe("no math");
    expect(normalizeLatexDelimiters("$already$")).toBe("$already$");
  });
});

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

describe("prepareMarkdownMath", () => {
  it("chains latex delimiters then display fence normalization", () => {
    const src = "\\(\\nu\\) in\n\n\\[R_g \\sim N^{\\nu}\\]";
    expect(prepareMarkdownMath(src)).toBe("$\\nu$ in\n\n$$\nR_g \\sim N^{\\nu}\n$$");
  });
});
