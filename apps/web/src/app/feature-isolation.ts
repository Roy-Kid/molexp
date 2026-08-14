/**
 * Feature-isolation scan for the three web-arch surfaces:
 * plugins/workflow, plugins/knowledge, plugins/molplot must not import each other.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export const FEATURE_PLUGIN_IDS = ["workflow", "knowledge", "molplot"] as const;
export type FeaturePluginId = (typeof FEATURE_PLUGIN_IDS)[number];

export type CrossFeatureHit = {
  file: string;
  from: FeaturePluginId;
  to: FeaturePluginId;
  specifier: string;
};

const IMPORT_FROM = /\b(?:import|export)\s+[\s\S]*?\bfrom\s*['"]([^'"]+)['"]/g;
const SIDE_EFFECT_IMPORT = /\bimport\s*['"]([^'"]+)['"]/g;
const DYNAMIC_IMPORT = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

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

export const featureTargetOf = (specifier: string, from: FeaturePluginId): FeaturePluginId | null => {
  for (const id of FEATURE_PLUGIN_IDS) {
    if (id === from) continue;
    if (specifier === `@/plugins/${id}` || specifier.startsWith(`@/plugins/${id}/`)) {
      return id;
    }
    if (specifier.includes(`/plugins/${id}/`) || specifier.endsWith(`/plugins/${id}`)) {
      return id;
    }
    if (specifier.includes(`/${id}/`) && specifier.startsWith(".")) {
      // relative hop such as ../../../knowledge/foo
      const parts = specifier.split("/");
      if (parts.includes(id)) return id;
    }
  }
  return null;
};

const walkTsFiles = (dir: string, acc: string[]): void => {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      walkTsFiles(full, acc);
      continue;
    }
    if (entry.endsWith(".ts") || entry.endsWith(".tsx")) acc.push(full);
  }
};

export const scanFeatureIsolation = (srcRoot: string): CrossFeatureHit[] => {
  const hits: CrossFeatureHit[] = [];
  for (const from of FEATURE_PLUGIN_IDS) {
    const files: string[] = [];
    walkTsFiles(join(srcRoot, "plugins", from), files);
    for (const file of files) {
      const source = readFileSync(file, "utf8");
      for (const specifier of collectImportSpecifiers(source)) {
        const to = featureTargetOf(specifier, from);
        if (to) {
          hits.push({
            file: relative(srcRoot, file).replace(/\\/g, "/"),
            from,
            to,
            specifier,
          });
        }
      }
    }
  }
  return hits;
};
