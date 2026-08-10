import { listFileTypeContributions } from "@/plugins/contribution-runtime";
import type { DiscoveredFile, FileMatchContext, FileTypeContribution } from "@/plugins/types";

const escapeRegExp = (input: string): string => {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
};

const compileGlob = (pattern: string): RegExp => {
  let regex = "";
  let i = 0;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === "*") {
      const next = pattern[i + 1];
      if (next === "*") {
        regex += ".*";
        i += 2;
      } else {
        regex += "[^/]*";
        i += 1;
      }
      continue;
    }
    if (ch === "?") {
      regex += "[^/]";
      i += 1;
      continue;
    }
    regex += escapeRegExp(ch);
    i += 1;
  }
  return new RegExp(`^${regex}$`, "i");
};

const globCache = new Map<string, RegExp>();

const getGlobRegex = (pattern: string): RegExp => {
  let cached = globCache.get(pattern);
  if (!cached) {
    cached = compileGlob(pattern);
    globCache.set(pattern, cached);
  }
  return cached;
};

export const matchesFile = (
  contribution: FileTypeContribution,
  file: FileMatchContext,
): boolean => {
  const { matcher } = contribution;
  if (matcher.patterns) {
    for (const pattern of matcher.patterns) {
      if (getGlobRegex(pattern).test(file.relPath) || getGlobRegex(pattern).test(file.name)) {
        return true;
      }
    }
  }
  if (matcher.matches?.(file)) {
    return true;
  }
  return false;
};

interface RawFileNode {
  name: string;
  relPath: string;
  type: string;
  size?: number | null;
  hasPreviewSidecar?: boolean;
  children?: RawFileNode[];
}

export const flattenFileNodes = (nodes: RawFileNode[] | undefined): FileMatchContext[] => {
  const out: FileMatchContext[] = [];
  const stack = [...(nodes ?? [])];
  while (stack.length > 0) {
    const node = stack.shift();
    if (!node) {
      continue;
    }
    if (node.type === "file") {
      out.push({
        name: node.name,
        relPath: node.relPath,
        size: node.size,
        type: node.type,
        hasPreviewSidecar: node.hasPreviewSidecar,
      });
    }
    if (node.children) {
      stack.push(...node.children);
    }
  }
  return out;
};

export interface DiscoveredPlugin {
  contribution: FileTypeContribution;
  files: DiscoveredFile[];
}

export const discoverPlugins = (
  contributions: FileTypeContribution[],
  files: FileMatchContext[],
): DiscoveredPlugin[] => {
  const out: DiscoveredPlugin[] = [];
  for (const contribution of contributions) {
    const matched: DiscoveredFile[] = [];
    for (const file of files) {
      if (matchesFile(contribution, file)) {
        matched.push({ ...file, matchedBy: contribution.id });
      }
    }
    if (matched.length > 0) {
      out.push({ contribution, files: matched });
    }
  }
  return out;
};

export const discoverPluginsForObject = (
  objectType: FileTypeContribution["objectType"],
  files: FileMatchContext[],
): DiscoveredPlugin[] => {
  return discoverPlugins(listFileTypeContributions(objectType), files);
};
