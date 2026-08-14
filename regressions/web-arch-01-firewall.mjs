#!/usr/bin/env node
/**
 * web-arch-01-firewall — same scanner as apps/web/src/plugins/import-guard.ts.
 * Expected offenders: [] (hard-coded). Run via `npx tsx regressions/web-arch-01-firewall.mjs`.
 */
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { scanShellVendorFirewall } from "../apps/web/src/plugins/import-guard.ts";

const srcRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../apps/web/src");
const expected = [];
const offenders = scanShellVendorFirewall(srcRoot);
assert.deepEqual(offenders, expected);
console.log("web-arch-01-firewall: ok");
