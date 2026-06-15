/**
 * RED build-output assertion: the Monaco editor must ship as an async,
 * on-demand chunk — NOT a script the browser fetches on initial page load.
 *
 * Once the editor plugin lazy-loads Monaco (`React.lazy` / dynamic import)
 * and the UI is rebuilt, the monaco chunk drops out of `index.html`'s
 * `<script>` set. Against the CURRENT (stale) build, `index.html` still
 * references the monaco chunk (`7948.*.js`), so this assertion is RED.
 *
 * If no build is present (`ui/dist` / `index.html` missing) — or no built
 * chunk contains the string `monaco` — the test skips cleanly rather than
 * failing; there is nothing meaningful to assert in those cases.
 *
 * Skip API used: rstest's `it.skipIf(condition)` (see `@rstest/core`
 * `TestAPI.skipIf`). The skip condition is computed once at module-eval
 * time from the filesystem, so there is no per-run `ctx.skip()` (rstest's
 * `TestContext` exposes no such method).
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// src/plugins/editor → src → ui → ui/dist
const distDir = resolve(__dirname, "../../../dist");
const indexHtmlPath = join(distDir, "index.html");
const jsDir = join(distDir, "static", "js");

const hasBuild = existsSync(distDir) && existsSync(indexHtmlPath) && existsSync(jsDir);

// Walk static/js recursively — rsbuild emits on-demand chunks under an
// `async/` subdirectory, so a flat readdir would miss the (post-lazy-load)
// monaco chunk and the test would skip instead of assert.
const listJsFilesRecursive = (dir: string): string[] => {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listJsFilesRecursive(full));
    } else if (entry.name.endsWith(".js")) {
      out.push(full);
    }
  }
  return out;
};

// Basenames of every built chunk whose contents reference monaco.
const findMonacoChunkBasenames = (): string[] => {
  if (!hasBuild) {
    return [];
  }
  return listJsFilesRecursive(jsDir)
    .filter((path) => readFileSync(path, "utf8").includes("monaco"))
    .map((path) => path.slice(path.lastIndexOf("/") + 1));
};

const monacoChunkBasenames = findMonacoChunkBasenames();

describe("monaco bundle split", () => {
  // Skip when there is no build, or no chunk carries monaco — nothing to assert.
  it.skipIf(!hasBuild || monacoChunkBasenames.length === 0)(
    "does not reference any monaco chunk as an initial page-load script in index.html",
    () => {
      const indexHtml = readFileSync(indexHtmlPath, "utf8");

      // Every monaco-carrying chunk must be loaded on demand, not wired into
      // the initial <script defer src=...> set of index.html.
      const referencedInitially = monacoChunkBasenames.filter((name) => indexHtml.includes(name));
      expect(referencedInitially).toEqual([]);
    },
  );
});
