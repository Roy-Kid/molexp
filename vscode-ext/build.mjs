/**
 * Build the MolExp Workflow Preview extension.
 *
 *   - extension host:  src/extension.ts  -> dist/extension.js   (node / cjs)
 *   - webview bundle:  webview/main.tsx  -> dist/webview.js      (browser / iife)
 *                      (+ dist/webview.css, the flowgram canvas CSS esbuild
 *                       collects from the JS graph)
 *   - webview theme:   webview/index.css -> dist/theme.css       (Tailwind v4
 *                       + shadcn tokens, via @tailwindcss/postcss)
 *
 * The webview reuses the molexp UI components directly: the `@/` alias points at
 * `../ui/src`, so `import { WorkflowPreview } from "@/components/workflow"`
 * resolves to the very files `molexp serve` ships. All third-party packages
 * (react, @flowgram.ai, radix, …) resolve from the hoisted `../node_modules`.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";
import tailwindcss from "@tailwindcss/postcss";
import postcss from "postcss";

const here = dirname(fileURLToPath(import.meta.url));
const uiSrc = resolve(here, "../ui/src");
const dist = resolve(here, "dist");
const watch = process.argv.includes("--watch");

if (!existsSync(uiSrc)) {
  console.error(`[build] cannot find molexp UI src at ${uiSrc}`);
  process.exit(1);
}
mkdirSync(dist, { recursive: true });

const nodePaths = [resolve(here, "../node_modules"), resolve(here, "../ui/node_modules")];

/** Shared esbuild settings. */
const common = {
  bundle: true,
  sourcemap: true,
  logLevel: "info",
  nodePaths,
};

const extensionConfig = {
  ...common,
  entryPoints: [resolve(here, "src/extension.ts")],
  outfile: resolve(dist, "extension.js"),
  platform: "node",
  format: "cjs",
  target: "node18",
  external: ["vscode"],
};

const webviewConfig = {
  ...common,
  entryPoints: [resolve(here, "webview/main.tsx")],
  outfile: resolve(dist, "webview.js"),
  platform: "browser",
  format: "iife",
  target: "es2020",
  jsx: "automatic",
  alias: { "@": uiSrc },
  loader: { ".css": "css" },
  define: { "process.env.NODE_ENV": '"production"' },
};

/** Compile the Tailwind theme stylesheet through @tailwindcss/postcss. */
async function buildTheme() {
  const input = resolve(here, "webview/index.css");
  const css = readFileSync(input, "utf8");
  const result = await postcss([tailwindcss()]).process(css, {
    from: input,
    to: resolve(dist, "theme.css"),
  });
  writeFileSync(resolve(dist, "theme.css"), result.css);
  console.log("[build] wrote dist/theme.css");
}

if (watch) {
  const ctxExt = await esbuild.context(extensionConfig);
  const ctxWeb = await esbuild.context(webviewConfig);
  await Promise.all([ctxExt.watch(), ctxWeb.watch()]);
  await buildTheme();
  console.log("[build] watching…");
} else {
  await Promise.all([esbuild.build(extensionConfig), esbuild.build(webviewConfig)]);
  await buildTheme();
  console.log("[build] done");
}
