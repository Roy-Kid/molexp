import { describe, expect, it } from "@rstest/core";

import { parseSlashQuery } from "../CommandPalette";

describe("parseSlashQuery", () => {
  it("does not open on plain text", () => {
    expect(parseSlashQuery("hello world")).toEqual({ open: false, query: "" });
  });

  it("opens on a bare slash with empty query", () => {
    expect(parseSlashQuery("/")).toEqual({ open: true, query: "" });
  });

  it("opens with a partial slash command and surfaces the query", () => {
    expect(parseSlashQuery("/pl")).toEqual({ open: true, query: "pl" });
  });

  it("trims leading whitespace before the slash", () => {
    expect(parseSlashQuery("   /audit")).toEqual({ open: true, query: "audit" });
  });

  it("does not open when the user has already typed an argument", () => {
    expect(parseSlashQuery("/plot metric=energy")).toEqual({ open: false, query: "" });
  });

  it("does not open mid-line slashes", () => {
    expect(parseSlashQuery("yes/no")).toEqual({ open: false, query: "" });
  });

  it("accepts hyphens and digits in the slug", () => {
    expect(parseSlashQuery("/plot-energy-2")).toEqual({
      open: true,
      query: "plot-energy-2",
    });
  });
});
