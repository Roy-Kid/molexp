#!/usr/bin/env node
/**
 * web-arch-02-layers — same scanner as apps/web/src/app/feature-isolation.ts.
 * Expected cross-feature imports: [] (hard-coded).
 * Run via `npx tsx regressions/web-arch-02-layers.mjs`.
 */
import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { scanFeatureIsolation } from "../apps/web/src/app/feature-isolation.ts";

const srcRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../apps/web/src");
const expected = [];
const hits = scanFeatureIsolation(srcRoot);
assert.deepEqual(hits, expected);
console.log("web-arch-02-layers: ok");
