import { describe, expect, it } from "@rstest/core";
import { collectFieldValues, type FormDocumentWire } from "./formDocument";

const DOC: FormDocumentWire = {
  title: "T",
  fields: [
    { kind: "text", id: "t", label: "Text", default: "a" },
    { kind: "textarea", id: "ta", label: "Area" },
    { kind: "number", id: "n", label: "Num", default: 1 },
    { kind: "boolean", id: "b", label: "Bool", default: true },
    {
      kind: "select",
      id: "s",
      label: "Sel",
      options: [{ value: "x", label: "X" }],
      default: "x",
    },
    {
      kind: "multi_select",
      id: "ms",
      label: "Multi",
      options: [{ value: "x", label: "X" }],
      default: ["x"],
    },
    {
      kind: "table",
      id: "tbl",
      label: "Table",
      columns: [{ id: "c", label: "C" }],
      default_rows: [{ c: "1" }],
    },
    { kind: "key_value", id: "kv", label: "KV", default: [{ key: "k", value: "v" }] },
    { kind: "markdown", id: "md", label: "MD", content: "**hi**", readonly: true },
    { kind: "artifact_ref", id: "ar", label: "Art", default: "id-1" },
  ],
};

describe("collectFieldValues", () => {
  it("collects editable defaults for all ten kinds without markdown", () => {
    const values = collectFieldValues(DOC);
    expect(values.t).toBe("a");
    expect(values.n).toBe(1);
    expect(values.b).toBe(true);
    expect(values.s).toBe("x");
    expect(values.ms).toEqual(["x"]);
    expect(values.ar).toBe("id-1");
    expect(values.md).toBeUndefined();
  });

  it("prefers overrides", () => {
    const values = collectFieldValues(DOC, { t: "over" });
    expect(values.t).toBe("over");
  });

  it("handles empty/missing docs", () => {
    expect(collectFieldValues(null)).toEqual({});
    expect(collectFieldValues({ fields: [] })).toEqual({});
  });
});

describe("FormField kind matrix", () => {
  it("documents all ten kinds in the fixture", () => {
    const kinds = new Set((DOC.fields ?? []).map((f) => f.kind));
    for (const k of [
      "text",
      "textarea",
      "number",
      "boolean",
      "select",
      "multi_select",
      "table",
      "key_value",
      "markdown",
      "artifact_ref",
    ]) {
      expect(kinds.has(k)).toBe(true);
    }
  });
});
