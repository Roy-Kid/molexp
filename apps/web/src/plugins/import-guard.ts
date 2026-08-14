/**
 * Source-scan vendor firewall for the apps/web shell.
 *
 * Walks `src/app/**` and `src/components/{ui,workbench}/**` and reports
 * static / type / re-export / dynamic imports whose specifier matches a
 * capability-vendor prefix. Comments are stripped first so a commented
 * `import("monaco-editor")` is not an offender.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export const FORBIDDEN_VENDOR_PREFIXES: readonly string[] = [
  "@milkdown",
  "@flowgram.ai",
  "@monaco-editor",
  "monaco-editor",
  "@molcrafts/molplot",
  "@molcrafts/molvis",
  "vega",
  "vega-lite",
  "vega-embed",
];

export const SHELL_SCAN_SEGMENTS: readonly string[] = [
  "app",
  "components/ui",
  "components/workbench",
];

const IMPORT_FROM = /\b(?:import|export)\s+[\s\S]*?\bfrom\s*['"]([^'"]+)['"]/g;
const SIDE_EFFECT_IMPORT = /\bimport\s*['"]([^'"]+)['"]/g;
const DYNAMIC_IMPORT = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

export type ShellVendorOffender = {
  file: string;
  specifier: string;
};

const stripComments = (source: string): string => {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
};

export const collectImportSpecifiers = (source: string): string[] => {
  const body = stripComments(source);
  const found: string[] = [];
  const pushAll = (re: RegExp): void => {
    re.lastIndex = 0;
    let match: RegExpExecArray | null = re.exec(body);
    while (match) {
      found.push(match[1]);
      match = re.exec(body);
    }
  };
  pushAll(IMPORT_FROM);
  pushAll(SIDE_EFFECT_IMPORT);
  pushAll(DYNAMIC_IMPORT);
  return found;
};

export const matchesForbiddenVendor = (specifier: string): boolean => {
  return FORBIDDEN_VENDOR_PREFIXES.some(
    (prefix) =>
      specifier === prefix ||
      specifier.startsWith(`${prefix}/`) ||
      specifier.startsWith(`${prefix}-`),
  );
};

export const forbiddenOffenders = (source: string): string[] => {
  return collectImportSpecifiers(source).filter(matchesForbiddenVendor);
};

const walkTsFiles = (dir: string, acc: string[]): void => {
  if (!existsSync(dir)) {
    return;
  }
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      walkTsFiles(full, acc);
      continue;
    }
    if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
      acc.push(full);
    }
  }
};

export const scanShellVendorFirewall = (srcRoot: string): ShellVendorOffender[] => {
  const files: string[] = [];
  for (const segment of SHELL_SCAN_SEGMENTS) {
    walkTsFiles(join(srcRoot, segment), files);
  }
  const hits: ShellVendorOffender[] = [];
  for (const file of files) {
    const source = readFileSync(file, "utf8");
    for (const specifier of forbiddenOffenders(source)) {
      hits.push({ file: relative(srcRoot, file).replace(/\\/g, "/"), specifier });
    }
  }
  return hits;
};
