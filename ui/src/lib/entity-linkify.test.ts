/**
 * entity-linkify unit tests (vision-loop-10).
 *
 * The contract is conservative by design (宁缺勿滥): only ids the workspace
 * snapshot actually knows become links; plausible-looking but unknown tokens
 * and anything inside code spans/fences stay byte-identical.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";
import type { WorkspaceSnapshot } from "@/app/types";
import { buildEntityLinkIndex, linkifyEntityTokens } from "@/lib/entity-linkify";

const RUN_PATH = "/projects/p1/experiments/e1/runs/a1b2c3d4";
const index = new Map<string, string>([["a1b2c3d4", RUN_PATH]]);

const countLinks = (text: string): number => (text.match(/\]\(/g) ?? []).length;

describe("linkifyEntityTokens", () => {
  it("wraps a known id in exactly one markdown link", () => {
    const out = linkifyEntityTokens("Run a1b2c3d4 finished.", index);
    expect(out).toBe(`Run [a1b2c3d4](${RUN_PATH}) finished.`);
    expect(countLinks(out)).toBe(1);
  });

  it("leaves an unknown 8-hex token untouched", () => {
    const text = "Run deadbeef finished.";
    expect(linkifyEntityTokens(text, index)).toBe(text);
  });

  it("never rewrites inline code spans or fenced blocks", () => {
    const text = "See `a1b2c3d4` and\n```\nrun a1b2c3d4\n```\ndone.";
    expect(linkifyEntityTokens(text, index)).toBe(text);
  });

  it("does not double-link an id that is already a markdown link", () => {
    const text = `[a1b2c3d4](${RUN_PATH})`;
    expect(linkifyEntityTokens(text, index)).toBe(text);
  });

  it("returns text byte-identical when the index is empty", () => {
    const text = "Run a1b2c3d4 finished.";
    expect(linkifyEntityTokens(text, new Map())).toBe(text);
  });
});

describe("buildEntityLinkIndex", () => {
  it("indexes runs, experiments, and projects onto their canonical paths", () => {
    const snapshot = {
      projects: [{ id: "p1" }],
      experiments: [{ id: "e1", projectId: "p1" }],
      runs: [{ id: "a1b2c3d4", projectId: "p1", experimentId: "e1" }],
    } as unknown as WorkspaceSnapshot;
    const built = buildEntityLinkIndex(snapshot);
    expect(built.get("a1b2c3d4")).toBe(RUN_PATH);
    expect(built.get("e1")).toBe("/projects/p1/experiments/e1");
    expect(built.get("p1")).toBe("/projects/p1");
    expect(built.has("ghost")).toBe(false);
  });
});

describe("conversation wiring (source contract)", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const read = (relative: string): string => readFileSync(resolve(here, relative), "utf-8");

  it("AgentViewer builds the index from the snapshot and threads it into turns", () => {
    const viewer = read("../app/renderers/AgentViewer.tsx");
    expect(viewer).toContain("buildEntityLinkIndex(snapshot)");
    expect(viewer).toContain("linkIndex={linkIndex}");
  });

  it("conversation prose runs through linkifyEntityTokens when an index is present", () => {
    const conversation = read("../app/renderers/agent/conversation.tsx");
    expect(conversation).toContain("linkifyEntityTokens(summary, linkIndex)");
    expect(conversation).toContain("linkifyEntityTokens(streamed.answer, linkIndex)");
  });
});
