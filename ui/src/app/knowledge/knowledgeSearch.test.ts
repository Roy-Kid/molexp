import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "@rstest/core";

/**
 * vision-loop-08 contracts: the Knowledge tab's search mode and the ⌘K
 * knowledge entries are wired to the ONE server search route. Source-contract
 * style (no jsdom in the rstest env — the ApprovalsInbox.test.ts convention).
 */

const read = (rel: string): string => readFileSync(join(__dirname, "..", "..", rel), "utf8");

describe("knowledge search surfaces", () => {
  const docTree = read("app/knowledge/DocTree.tsx");
  const api = read("app/state/api.ts");
  const generated = read("api/generated/services/KnowledgeService.ts");

  it("DocTree has a search mode consuming the search API", () => {
    expect(docTree).toContain("searchKnowledge");
    expect(docTree).toContain("Search notes");
    // Non-empty query switches to the flat hit list; entity hits route to
    // their entity pages, note hits to the knowledge surface.
    expect(docTree).toContain("selectionForSearchHit(hit)");
    // The truncated flag is surfaced, never silently dropped.
    expect(docTree).toContain("searchTruncated");
  });

  it("the api wrapper forwards to the generated search operation", () => {
    expect(api).toContain("searchKnowledge");
    expect(api).toContain("searchKnowledgeApiKnowledgeSearchGet");
    expect(generated).toContain("/api/knowledge/search");
  });
});

describe("selectionForSearchHit", () => {
  // Pure routing helper — imported directly (node env, no jsdom needed).
  it("routes entity hits to entity pages and notes to knowledge", async () => {
    const { selectionForSearchHit } = await import("./searchHitSelection");
    expect(
      selectionForSearchHit({
        path: "projects/p1/experiments/e1/runs/run-abc123",
        type: "workspace.run",
      }),
    ).toEqual({ objectType: "run", objectId: "abc123" });
    expect(
      selectionForSearchHit({ path: "projects/p1/experiments/e1", type: "workspace.experiment" }),
    ).toEqual({ objectType: "experiment", objectId: "e1" });
    expect(selectionForSearchHit({ path: "projects/p1", type: "workspace.project" })).toEqual({
      objectType: "project",
      objectId: "p1",
    });
    expect(selectionForSearchHit({ path: "protocols/gel-prep", type: "okf.note" })).toEqual({
      objectType: "knowledge",
      objectId: "protocols/gel-prep",
    });
    // An entity hit with an unparseable identity path degrades to knowledge,
    // never a broken entity link.
    expect(selectionForSearchHit({ path: "weird/place", type: "workspace.run" })).toEqual({
      objectType: "knowledge",
      objectId: "weird/place",
    });
  });
});

describe("⌘K knowledge entries", () => {
  const catalog = read("app/entities/catalog.ts");
  const palette = read("app/entities/GlobalCommandPalette.tsx");
  const kinds = read("app/entities/kinds.ts");
  const paths = read("app/entities/paths.ts");

  it("buildCatalog emits knowledge entries searchable by title/name", () => {
    expect(catalog).toContain("knowledgeDocs: KnowledgeDocEntry[]");
    expect(catalog).toContain('kind: "knowledge", id: doc.relPath');
  });

  it("the palette feeds knowledge docs into the catalog", () => {
    expect(palette).toContain("listKnowledge()");
    expect(palette).toContain("buildCatalog(snapshot, knowledgeDocs)");
  });

  it("the knowledge kind navigates to the Knowledge tab doc path", () => {
    expect(kinds).toContain('kind: "knowledge"');
    expect(paths).toContain('case "knowledge"');
  });
});
