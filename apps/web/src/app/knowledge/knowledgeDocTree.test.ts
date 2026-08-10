import { describe, expect, it } from "@rstest/core";
import {
  buildDocTree,
  buildOutline,
  type DocEntry,
  type DocTreeNode,
} from "@/app/knowledge/knowledgeDocTree";

/** Depth-first search for the first node whose `relPath` matches. */
const findByRelPath = (nodes: DocTreeNode[], relPath: string): DocTreeNode | null => {
  for (const node of nodes) {
    if (node.relPath === relPath) return node;
    const hit = findByRelPath(node.children, relPath);
    if (hit) return hit;
  }
  return null;
};

/** Depth-first search for the first node whose `id` matches. */
const findById = (nodes: DocTreeNode[], id: string): DocTreeNode | null => {
  for (const node of nodes) {
    if (node.id === id) return node;
    const hit = findById(node.children, id);
    if (hit) return hit;
  }
  return null;
};

describe("buildDocTree (ac-001)", () => {
  it("nests a child Note beneath its parent by relPath segments (kb/a/b under kb/a)", () => {
    const entries: DocEntry[] = [
      { relPath: "kb/a", name: "A" },
      { relPath: "kb/a/b", name: "B" },
    ];

    const tree = buildDocTree(entries);

    const parent = findByRelPath(tree, "kb/a");
    const child = findByRelPath(tree, "kb/a/b");
    expect(parent).not.toBeNull();
    expect(child).not.toBeNull();
    // The child must live inside the parent's subtree, never as a sibling.
    expect(findByRelPath(parent?.children ?? [], "kb/a/b")).not.toBeNull();
  });

  it("presents a dedicated knowledge-base group AND one group per entity with docs", () => {
    const entries: DocEntry[] = [
      { relPath: "kb/a", name: "A" },
      { relPath: "projects/p1/note-x", name: "X" },
      { relPath: "projects/p1/experiments/e1/note-y", name: "Y" },
      { relPath: "projects/p1/experiments/e1/runs/run-r1/note-z", name: "Z" },
    ];

    const tree = buildDocTree(entries);

    // Every top-level node is a group.
    expect(tree.every((n) => n.kind === "group")).toBe(true);

    // The dedicated knowledge-base group holds the root-bundle doc.
    const kbGroup = tree.find((n) => n.entity === undefined);
    expect(kbGroup).toBeDefined();
    expect(findByRelPath(kbGroup?.children ?? [], "kb/a")).not.toBeNull();

    // One group per attached entity (project / experiment / run).
    const project = tree.find((n) => n.entity?.kind === "project" && n.entity.id === "p1");
    const experiment = tree.find((n) => n.entity?.kind === "experiment" && n.entity.id === "e1");
    const run = tree.find((n) => n.entity?.kind === "run" && n.entity.id === "r1");
    expect(project).toBeDefined();
    expect(experiment).toBeDefined();
    expect(run).toBeDefined();

    // The doc lands inside its owning entity's group.
    expect(findByRelPath(project?.children ?? [], "projects/p1/note-x")).not.toBeNull();
    expect(
      findByRelPath(experiment?.children ?? [], "projects/p1/experiments/e1/note-y"),
    ).not.toBeNull();
    expect(
      findByRelPath(run?.children ?? [], "projects/p1/experiments/e1/runs/run-r1/note-z"),
    ).not.toBeNull();
  });

  it("yields an empty tree for empty input", () => {
    expect(buildDocTree([])).toEqual([]);
  });

  it("renders a single root doc as one leaf node with no children", () => {
    const tree = buildDocTree([{ relPath: "solo", name: "Solo" }]);

    const kbGroup = tree.find((n) => n.entity === undefined);
    expect(kbGroup).toBeDefined();
    const doc = findByRelPath(tree, "solo");
    expect(doc?.kind).toBe("doc");
    expect(doc?.children).toEqual([]);
  });

  it("preserves deep nesting order and parent/child relationships", () => {
    const entries: DocEntry[] = [
      { relPath: "kb/a/b/c", name: "C" },
      { relPath: "kb/a", name: "A" },
      { relPath: "kb/a/b", name: "B" },
    ];

    const tree = buildDocTree(entries);

    const a = findByRelPath(tree, "kb/a");
    const b = findByRelPath(a?.children ?? [], "kb/a/b");
    const c = findByRelPath(b?.children ?? [], "kb/a/b/c");
    expect(a?.kind).toBe("doc");
    expect(b?.kind).toBe("doc");
    expect(c?.kind).toBe("doc");
    // The intermediate "kb" segment (no Note of its own) is a dir container.
    const kbDir = findById(tree, "kb");
    expect(kbDir?.kind).toBe("dir");
  });
});

describe("buildOutline (ac-002)", () => {
  it("returns ordered H1–H3 headings in document order", () => {
    const md = ["# Title", "intro", "## Section", "### Sub", "## Another"].join("\n");

    expect(buildOutline(md)).toEqual([
      { level: 1, text: "Title", slug: "title" },
      { level: 2, text: "Section", slug: "section" },
      { level: 3, text: "Sub", slug: "sub" },
      { level: 2, text: "Another", slug: "another" },
    ]);
  });

  it("skips H4+ headings", () => {
    const md = ["# A", "#### D", "##### E", "###### F", "## B"].join("\n");

    expect(buildOutline(md).map((h) => h.text)).toEqual(["A", "B"]);
    expect(buildOutline(md).every((h) => h.level <= 3)).toBe(true);
  });

  it("ignores '#' lines inside fenced code blocks", () => {
    const md = [
      "# Real",
      "```py",
      "# not a heading",
      "## also not a heading",
      "```",
      "## After",
    ].join("\n");

    expect(buildOutline(md)).toEqual([
      { level: 1, text: "Real", slug: "real" },
      { level: 2, text: "After", slug: "after" },
    ]);
  });

  it("returns an empty outline for a heading-free document", () => {
    expect(buildOutline("plain text\nmore text\n")).toEqual([]);
    expect(buildOutline("")).toEqual([]);
  });

  it("slugifies punctuation and whitespace", () => {
    expect(buildOutline("## Hello, World!")).toEqual([
      { level: 2, text: "Hello, World!", slug: "hello-world" },
    ]);
  });
});
