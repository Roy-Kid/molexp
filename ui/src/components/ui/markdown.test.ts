/**
 * Math-rendering contract for every markdown text surface.
 *
 * Rstest runs in a node environment without jsdom (rstest.config.ts:
 * testEnvironment: "node"), so — per repo convention (see
 * RunMetricsView.test.tsx) — the JSX wiring is asserted by reading the
 * component sources: each markdown surface must parse `$...$` / `$$...$$`
 * through remark-math and render it with rehype-katex (KaTeX CSS imported),
 * and the Milkdown note editor must mount the math plugin. Without this
 * wiring, LLM-generated reports (`r_min = 2^{1/6}\sigma`) and knowledge
 * notes show raw dollar-delimited source instead of formulas.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const read = (rel: string): string => readFileSync(resolve(__dirname, rel), "utf-8");

describe("MarkdownContent math wiring", () => {
  const source = read("./markdown.tsx");

  it("parses math and renders it with KaTeX", () => {
    expect(source).toContain("remark-math");
    expect(source).toContain("rehype-katex");
    expect(source).toContain("remarkMath");
    expect(source).toContain("rehypeKatex");
  });

  it("ships the KaTeX stylesheet", () => {
    expect(source).toContain('import "katex/dist/katex.min.css"');
  });

  it("styles display math with safe horizontal overflow", () => {
    expect(source).toContain("katex-display");
    expect(source).toContain("overflow-x-auto");
  });

  it("normalizes whole-line $$…$$ into display math before parsing", () => {
    // remark-math parses single-line `$$…$$` as INLINE math (verified against
    // the live pipeline) — the text must pass through normalizeDisplayMath or
    // block formulas render squashed and un-centered (lj-theory regression).
    expect(source).toContain("normalizeDisplayMath(text)");
  });
});

describe("MarkdownPreview math wiring", () => {
  const source = read("../previews/MarkdownPreview.tsx");

  it("parses math and renders it with KaTeX", () => {
    expect(source).toContain("remarkMath");
    expect(source).toContain("rehypeKatex");
    expect(source).toContain('import "katex/dist/katex.min.css"');
  });

  it("normalizes whole-line $$…$$ into display math before parsing", () => {
    expect(source).toContain("normalizeDisplayMath(content)");
  });
});

describe("NoteEditor math wiring", () => {
  const source = read("../../app/renderers/knowledge/NoteEditor.tsx");

  it("mounts the Milkdown math plugin with the KaTeX stylesheet", () => {
    expect(source).toContain('from "@milkdown/plugin-math"');
    expect(source).toContain(".use(math)");
    expect(source).toContain('import "katex/dist/katex.min.css"');
  });
});
