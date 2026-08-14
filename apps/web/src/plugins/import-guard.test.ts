import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";
import {
  collectImportSpecifiers,
  forbiddenOffenders,
  SHELL_SCAN_SEGMENTS,
  scanShellVendorFirewall,
} from "./import-guard";

const srcRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("vendor firewall", () => {
  it("collects static, type, re-export, and dynamic import specifiers", () => {
    const source = `
      import { x } from "@milkdown/kit/core";
      import type { LineChartConfig } from "@molcrafts/molplot";
      export { y } from "vega-lite";
      const m = await import("@flowgram.ai/free-layout-editor");
      const n = import("@monaco-editor/react");
      import("@molcrafts/molvis-core");
    `;
    expect(collectImportSpecifiers(source)).toEqual([
      "@milkdown/kit/core",
      "@molcrafts/molplot",
      "vega-lite",
      "@flowgram.ai/free-layout-editor",
      "@monaco-editor/react",
      "@molcrafts/molvis-core",
    ]);
  });

  it("flags planted static and dynamic vendor imports", () => {
    expect(forbiddenOffenders('import { x } from "@milkdown/kit/core";')).toEqual([
      "@milkdown/kit/core",
    ]);
    expect(forbiddenOffenders('await import("@flowgram.ai/free-layout-editor");')).toEqual([
      "@flowgram.ai/free-layout-editor",
    ]);
    expect(
      forbiddenOffenders('import type { LineChartConfig } from "@molcrafts/molplot";'),
    ).toEqual(["@molcrafts/molplot"]);
    expect(forbiddenOffenders('import("@monaco-editor/react");')).toEqual(["@monaco-editor/react"]);
    expect(forbiddenOffenders('import("@molcrafts/molvis-core");')).toEqual([
      "@molcrafts/molvis-core",
    ]);
  });

  it("ignores commented vendor imports", () => {
    expect(forbiddenOffenders('// import("monaco-editor")\nconst ok = 1;')).toEqual([]);
    expect(forbiddenOffenders('/* import("@milkdown/kit") */\nexport const n = 1;')).toEqual([]);
  });

  it("walks only the shell trees", () => {
    expect(SHELL_SCAN_SEGMENTS).toEqual(["app", "components/ui", "components/workbench"]);
    const hits = scanShellVendorFirewall(srcRoot);
    expect(hits.every((hit) => !hit.file.startsWith("plugins/"))).toBe(true);
    expect(hits.every((hit) => !hit.file.startsWith("components/workflow/"))).toBe(true);
  });

  it("reports zero shell vendor offenders", () => {
    expect(scanShellVendorFirewall(srcRoot)).toEqual([]);
  });
});
