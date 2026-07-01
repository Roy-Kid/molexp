import { describe, expect, it } from "@rstest/core";
import { buildNoteDocUpdate, isDirty, normalizeMarkdown } from "../noteDraft";

describe("normalizeMarkdown", () => {
  it("is idempotent (normalize(normalize(x)) === normalize(x))", () => {
    const samples = [
      "hello world",
      "line one \nline two\t\n",
      "# Heading\n\n- a\n- b\n",
      "trailing\r\nwindows\r\nendings\r\n",
      "",
      "  \n  \n",
      "mixed\r\nendings \r  and \ttrailing   ",
    ];
    for (const sample of samples) {
      const once = normalizeMarkdown(sample);
      expect(normalizeMarkdown(once)).toBe(once);
    }
  });

  it("keeps an empty body ('') stable", () => {
    expect(normalizeMarkdown("")).toBe("");
    expect(normalizeMarkdown(normalizeMarkdown(""))).toBe("");
  });

  it("normalizes CRLF to LF and trims per-line trailing whitespace", () => {
    expect(normalizeMarkdown("a  \r\nb\t\r\n")).toBe("a\nb");
  });
});

describe("isDirty", () => {
  it("is false when comparing a body to itself", () => {
    const body = "# Note\n\nSome content.";
    expect(isDirty(body, body)).toBe(false);
  });

  it("is true after a real content edit", () => {
    const original = "# Note\n\nSome content.";
    const edited = "# Note\n\nSome edited content.";
    expect(isDirty(original, edited)).toBe(true);
  });

  it("is false for a trailing-whitespace-only edit (normalized-equal)", () => {
    const original = "# Note\n\nSome content.";
    const edited = "# Note   \n\nSome content.   \n";
    expect(isDirty(original, edited)).toBe(false);
  });

  it("treats an empty body edited to whitespace-only as not dirty", () => {
    expect(isDirty("", "   \n  \n")).toBe(false);
  });
});

describe("buildNoteDocUpdate", () => {
  it("returns a payload carrying the path and normalized body", () => {
    const update = buildNoteDocUpdate("notes/intro", "hello   \r\nworld\r\n");
    expect(update).toEqual({ path: "notes/intro", body: "hello\nworld" });
  });
});
