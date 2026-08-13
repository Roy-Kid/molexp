/**
 * Pure, React-free, IO-free builders for the Knowledge document shell:
 *
 * - {@link buildDocTree} recovers parent/child nesting from Note `relPath`
 *   segments and buckets the top level into a dedicated knowledge-base group
 *   (workspace-root bundle docs) plus one group per attached entity
 *   (project / experiment / run).
 * - {@link buildOutline} extracts a document's H1–H3 headings, in order,
 *   ignoring H4+ and any `#` lines inside fenced code blocks.
 *
 * These are the sole automatable units of the 04 knowledge-tree feature; the
 * surrounding tree / panel components are non-binding UI verification.
 */

export interface DocEntry {
  /** The Note's bundle-relative identity path (its directory). */
  relPath: string;
  /** Display name for the Note. */
  name: string;
}

export type DocEntityKind = "project" | "experiment" | "run";

export type DocTreeNodeKind = "group" | "dir" | "doc";

export interface DocTreeNode {
  /** Stable id: the accumulated relPath for docs/dirs, or `group:*` for groups. */
  id: string;
  /** Display label. */
  name: string;
  kind: DocTreeNodeKind;
  /** The Note's bundle-relative identity path — set only for `kind === "doc"`. */
  relPath: string | null;
  /** For entity groups: the owning entity kind + id. Absent on the KB group. */
  entity?: { kind: DocEntityKind; id: string };
  children: DocTreeNode[];
}

export interface OutlineHeading {
  level: 1 | 2 | 3;
  text: string;
  slug: string;
}

export const KB_GROUP_ID = "group:knowledge-base";
const KB_GROUP_NAME = "Knowledge base";

interface Owner {
  groupId: string;
  groupName: string;
  entity?: { kind: DocEntityKind; id: string };
  /** Number of leading path segments consumed by the entity prefix. */
  prefixLen: number;
}

/** Drop the mandatory `run-` directory prefix to recover the bare run id. */
const stripRunPrefix = (segment: string): string =>
  segment.startsWith("run-") ? segment.slice("run-".length) : segment;

/**
 * Classify a Note by its relPath: which top-level group owns it and how many
 * leading segments are the entity-container prefix. A relPath under
 * `projects/…[/experiments/…[/runs/run-…]]` belongs to the deepest such entity;
 * everything else is a root-bundle knowledge-base doc.
 */
const classify = (segments: string[]): Owner => {
  if (segments[0] === "projects" && segments.length >= 2) {
    const projectId = segments[1];
    if (segments[2] === "experiments" && segments.length >= 4) {
      const experimentId = segments[3];
      if (segments[4] === "runs" && segments.length >= 6) {
        const runId = stripRunPrefix(segments[5]);
        return {
          groupId: `group:run:${runId}`,
          groupName: runId,
          entity: { kind: "run", id: runId },
          prefixLen: 6,
        };
      }
      return {
        groupId: `group:experiment:${experimentId}`,
        groupName: experimentId,
        entity: { kind: "experiment", id: experimentId },
        prefixLen: 4,
      };
    }
    return {
      groupId: `group:project:${projectId}`,
      groupName: projectId,
      entity: { kind: "project", id: projectId },
      prefixLen: 2,
    };
  }
  return { groupId: KB_GROUP_ID, groupName: KB_GROUP_NAME, prefixLen: 0 };
};

/** Insert one Note into its group, materializing intermediate dir nodes. */
const insert = (group: DocTreeNode, entry: DocEntry, prefixLen: number): void => {
  const segments = entry.relPath.split("/");
  const remaining = segments.slice(prefixLen);

  // A Note whose relPath is exactly the entity container (no sub-path) attaches
  // directly under the group as a leaf.
  if (remaining.length === 0) {
    upsertDoc(group.children, entry.relPath, entry.relPath, entry.name);
    return;
  }

  let level = group.children;
  for (let i = 0; i < remaining.length; i += 1) {
    const nodeRelPath = segments.slice(0, prefixLen + i + 1).join("/");
    const isLeaf = i === remaining.length - 1;
    let node = level.find((n) => n.id === nodeRelPath);
    if (!node) {
      node = {
        id: nodeRelPath,
        name: isLeaf ? entry.name : remaining[i],
        kind: isLeaf ? "doc" : "dir",
        relPath: isLeaf ? entry.relPath : null,
        children: [],
      };
      level.push(node);
    } else if (isLeaf) {
      // A previously-created intermediate dir is now confirmed to be a Note.
      node.kind = "doc";
      node.relPath = entry.relPath;
      node.name = entry.name;
    }
    level = node.children;
  }
};

/** Add or promote a leaf doc node in `children` (idempotent on id). */
const upsertDoc = (children: DocTreeNode[], id: string, relPath: string, name: string): void => {
  const existing = children.find((n) => n.id === id);
  if (existing) {
    existing.kind = "doc";
    existing.relPath = relPath;
    existing.name = name;
    return;
  }
  children.push({ id, name, kind: "doc", relPath, children: [] });
};

const ENTITY_RANK: Record<DocEntityKind, number> = { project: 0, experiment: 1, run: 2 };

/** Sort a node's children in place (dirs before docs, each alphabetical). */
const sortChildren = (nodes: DocTreeNode[]): void => {
  nodes.sort((a, b) => {
    if (a.kind !== b.kind) {
      // dirs group above sibling docs for a stable, readable tree.
      if (a.kind === "dir") return -1;
      if (b.kind === "dir") return 1;
    }
    return a.name.localeCompare(b.name);
  });
  for (const node of nodes) sortChildren(node.children);
};

/**
 * Assemble a nested document tree from a flat list of Notes. The top level is a
 * dedicated knowledge-base group (root-bundle docs) plus one group per entity
 * (project / experiment / run) that has attached docs. Nesting within a group is
 * recovered from the Notes' relPath segments. Empty input yields an empty tree.
 */
export const buildDocTree = (entries: DocEntry[]): DocTreeNode[] => {
  const groups = new Map<string, DocTreeNode>();

  for (const entry of entries) {
    const segments = entry.relPath.split("/").filter(Boolean);
    if (segments.length === 0) continue;
    const owner = classify(segments);
    let group = groups.get(owner.groupId);
    if (!group) {
      group = {
        id: owner.groupId,
        name: owner.groupName,
        kind: "group",
        relPath: null,
        ...(owner.entity ? { entity: owner.entity } : {}),
        children: [],
      };
      groups.set(owner.groupId, group);
    }
    insert(group, { ...entry, relPath: segments.join("/") }, owner.prefixLen);
  }

  const ordered = [...groups.values()].sort((a, b) => {
    // The KB group leads; entity groups follow, ranked by kind then id.
    if (a.entity === undefined) return b.entity === undefined ? 0 : -1;
    if (b.entity === undefined) return 1;
    if (a.entity.kind !== b.entity.kind) {
      return ENTITY_RANK[a.entity.kind] - ENTITY_RANK[b.entity.kind];
    }
    return a.entity.id.localeCompare(b.entity.id);
  });

  for (const group of ordered) sortChildren(group.children);
  return ordered;
};

const FENCE_RE = /^[ \t]{0,3}(?:`{3,}|~{3,})/;
const HEADING_RE = /^[ \t]{0,3}(#{1,6})[ \t]+(.*)$/;

/** GitHub-flavored slug: lowercase, punctuation stripped, spaces → hyphens. */
const slugify = (text: string): string =>
  text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");

/**
 * Extract a document's H1–H3 headings in document order. H4+ headings are
 * skipped, and any `#` line inside a fenced code block (``` or ~~~) is ignored.
 */
export const buildOutline = (markdown: string): OutlineHeading[] => {
  const headings: OutlineHeading[] = [];
  let inFence = false;

  for (const line of markdown.split("\n")) {
    if (FENCE_RE.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;

    const match = line.match(HEADING_RE);
    if (!match) continue;
    const level = match[1].length;
    if (level > 3) continue;

    // Strip an optional ATX closing sequence (`## Heading ##`).
    const text = match[2].replace(/[ \t]+#+[ \t]*$/, "").trim();
    if (!text) continue;
    headings.push({ level: level as 1 | 2 | 3, text, slug: slugify(text) });
  }

  return headings;
};
