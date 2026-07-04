import { describe, expect, it } from "@rstest/core";
import { SLASH_COMMANDS, slashCommandMarkdown } from "@/app/renderers/knowledge/slashCommands";

describe("slashCommandMarkdown (ac-001)", () => {
  it("maps block commands to their markdown prefixes", () => {
    expect(slashCommandMarkdown("heading1")).toBe("# ");
    expect(slashCommandMarkdown("heading3")).toBe("### ");
    expect(slashCommandMarkdown("bulletList")).toBe("- ");
    expect(slashCommandMarkdown("orderedList")).toBe("1. ");
    expect(slashCommandMarkdown("checkbox")).toBe("- [ ] ");
    expect(slashCommandMarkdown("quote")).toBe("> ");
    expect(slashCommandMarkdown("divider")).toBe("---\n");
  });

  it("maps a code block to a valid fenced block skeleton", () => {
    const snippet = slashCommandMarkdown("codeBlock");
    // Must open and close a fenced code block so markdown/GFM parses it.
    expect(snippet.startsWith("```")).toBe(true);
    expect(snippet.split("```").length - 1).toBe(2);
  });

  it("maps a table to a parseable GFM table skeleton", () => {
    const snippet = slashCommandMarkdown("table");
    // A GFM table needs a header row and a delimiter row of dashes.
    expect(snippet).toContain("| --- |");
    expect(snippet.split("\n").length).toBeGreaterThanOrEqual(3);
  });

  it("returns a safe empty default for an unknown id and never throws", () => {
    expect(() => slashCommandMarkdown("does-not-exist")).not.toThrow();
    expect(slashCommandMarkdown("does-not-exist")).toBe("");
    expect(slashCommandMarkdown("")).toBe("");
  });

  it("gives every advertised command a non-empty markdown snippet", () => {
    expect(SLASH_COMMANDS.length).toBeGreaterThan(0);
    for (const command of SLASH_COMMANDS) {
      expect(slashCommandMarkdown(command.id).length).toBeGreaterThan(0);
    }
  });

  it("advertises a unique id, label, and keyword set per command", () => {
    const ids = SLASH_COMMANDS.map((command) => command.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const command of SLASH_COMMANDS) {
      expect(command.label.length).toBeGreaterThan(0);
      expect(Array.isArray(command.keywords)).toBe(true);
    }
  });
});
