/**
 * Tokenizer contract for the Deliverables code panels (see highlight.ts):
 * python + yaml token classes for review readability, plus the lossless
 * invariant — concatenating token texts always reproduces the input.
 */

import { describe, expect, it } from "@rstest/core";

import { type HighlightToken, highlightCode } from "./highlight";

const reassemble = (tokens: HighlightToken[]): string => tokens.map((t) => t.text).join("");

const kindsOf = (tokens: HighlightToken[], kind: string): string[] =>
  tokens.filter((t) => t.kind === kind).map((t) => t.text);

describe("highlightCode python", () => {
  const src = [
    "# build the workflow",
    "@wf.task(depends_on=['grid'])",
    'def compute(ctx, sigma: float = 1.0) -> dict:',
    '    """Return the LJ minimum."""',
    "    label = 'lj'  # inline note",
    "    return {'r_min': 2 ** (1 / 6) * sigma, 'tol': 1e-3}",
  ].join("\n");
  const tokens = highlightCode(src, "python");

  it("is lossless (token texts reassemble the source)", () => {
    expect(reassemble(tokens)).toBe(src);
  });

  it("classifies comments, including trailing ones", () => {
    expect(kindsOf(tokens, "comment")).toEqual(["# build the workflow", "# inline note"]);
  });

  it("classifies keywords", () => {
    expect(kindsOf(tokens, "keyword")).toContain("def");
    expect(kindsOf(tokens, "keyword")).toContain("return");
  });

  it("classifies decorators", () => {
    expect(kindsOf(tokens, "decorator")).toEqual(["@wf.task"]);
  });

  it("classifies strings — quoted and docstring — without leaking comments into them", () => {
    const strings = kindsOf(tokens, "string");
    expect(strings).toContain('"""Return the LJ minimum."""');
    expect(strings).toContain("'lj'");
    expect(strings).toContain("'r_min'");
  });

  it("classifies numbers including scientific notation", () => {
    expect(kindsOf(tokens, "number")).toContain("1e-3");
  });

  it("keeps a # inside a string literal as part of the string", () => {
    const t = highlightCode("x = 'a # b'", "python");
    expect(kindsOf(t, "string")).toEqual(["'a # b'"]);
    expect(kindsOf(t, "comment")).toEqual([]);
    expect(reassemble(t)).toBe("x = 'a # b'");
  });

  it("spans multi-line triple-quoted strings", () => {
    const t = highlightCode('doc = """line one\nline two"""\nx = 1', "python");
    expect(kindsOf(t, "string")).toEqual(['"""line one\nline two"""']);
    expect(reassemble(t)).toBe('doc = """line one\nline two"""\nx = 1');
  });
});

describe("highlightCode yaml", () => {
  const src = [
    "# experiment spec",
    "objective: scan the LJ minimum",
    "tolerance: 1.0e-3",
    "grounded: true",
    "tasks:",
    "  - id: compute_energy",
    '    purpose: "V(r) curve"  # per-task',
  ].join("\n");
  const tokens = highlightCode(src, "yaml");

  it("is lossless (token texts reassemble the source)", () => {
    expect(reassemble(tokens)).toBe(src);
  });

  it("classifies mapping keys, including nested list-item keys", () => {
    const keys = kindsOf(tokens, "key");
    expect(keys).toEqual(["objective", "tolerance", "grounded", "tasks", "id", "purpose"]);
  });

  it("classifies comments only at line start or after whitespace", () => {
    expect(kindsOf(tokens, "comment")).toEqual(["# experiment spec", "# per-task"]);
  });

  it("classifies quoted strings and scalar literals", () => {
    expect(kindsOf(tokens, "string")).toEqual(['"V(r) curve"']);
    expect(kindsOf(tokens, "keyword")).toEqual(["true"]);
    expect(kindsOf(tokens, "number")).toContain("1.0e-3");
  });

  it("does not treat an anchor # inside a value as a comment", () => {
    const t = highlightCode("url: https://example.org/page#frag", "yaml");
    expect(kindsOf(t, "comment")).toEqual([]);
    expect(reassemble(t)).toBe("url: https://example.org/page#frag");
  });
});

describe("highlightCode fallback", () => {
  it("returns one plain token for unknown languages", () => {
    expect(highlightCode("{\"a\": 1}", "json")).toEqual([{ text: "{\"a\": 1}", kind: "plain" }]);
    expect(highlightCode("text")).toEqual([{ text: "text", kind: "plain" }]);
  });

  it("returns no tokens for empty input", () => {
    expect(highlightCode("", "python")).toEqual([]);
    expect(highlightCode("")).toEqual([]);
  });
});
