// ─────────────────────────────────────────────────────────────────────────────
// entity-linkify — conservative entity links in agent prose (vision-loop-10).
//
// Given the workspace snapshot's KNOWN ids (runs / experiments / projects),
// wrap exact-token matches in agent conversation markdown with links to the
// entity's canonical app path. Matching only against known ids means zero
// false links (宁缺勿滥): an 8-hex token that is not in the snapshot is left
// untouched. Code spans/fences are never rewritten.
// ─────────────────────────────────────────────────────────────────────────────

import { entityPath, runPath } from "@/app/entities/paths";
import type { WorkspaceSnapshot } from "@/app/types";

/** id → canonical app path, for every entity the snapshot knows. */
export const buildEntityLinkIndex = (snapshot: WorkspaceSnapshot): Map<string, string> => {
  const index = new Map<string, string>();
  for (const run of snapshot.runs) {
    index.set(run.id, runPath(run.projectId, run.experimentId, run.id));
  }
  for (const experiment of snapshot.experiments) {
    const path = entityPath({ kind: "experiment", id: experiment.id }, snapshot);
    if (path) index.set(experiment.id, path);
  }
  for (const project of snapshot.projects) {
    index.set(project.id, `/projects/${encodeURIComponent(project.id)}`);
  }
  return index;
};

// A candidate token: word characters plus dash (run ids are 8-hex or slugs).
const TOKEN = /[A-Za-z0-9][\w-]*/g;

// Split out code segments (fences + inline spans) — never rewritten.
const CODE_SEGMENT = /(```[\s\S]*?```|`[^`]*`)/g;

const linkifySegment = (segment: string, index: Map<string, string>): string =>
  segment.replace(TOKEN, (token, offset: number) => {
    const path = index.get(token);
    if (!path) return token;
    // Already inside a markdown link target/label? Skip conservatively.
    const before = segment.slice(Math.max(0, offset - 2), offset);
    if (before.includes("[") || before.includes("](") || before.includes("/")) return token;
    return `[${token}](${path})`;
  });

/**
 * Wrap exact known-id tokens in *text* with markdown links to their entity
 * pages. Only ids present in *index* are linked; everything else — including
 * plausible-looking but unknown ids — is left byte-identical.
 */
export const linkifyEntityTokens = (text: string, index: Map<string, string>): string => {
  if (index.size === 0 || !text) return text;
  return text
    .split(CODE_SEGMENT)
    .map((segment, i) => (i % 2 === 1 ? segment : linkifySegment(segment, index)))
    .join("");
};
