import { describe, expect, it } from "@rstest/core";
import type { WorkspaceTreeNode } from "@/app/types";
import {
  HttpWorkspaceFs,
  mapDirent,
  mergeTreeChildren,
  treeRootFromListing,
} from "@/lib/workspace-fs";

describe("mapDirent depth budget", () => {
  it("marks leaf dirs not-loaded when budget is 0", () => {
    const d = mapDirent(
      { name: "projects", path: "/ws/projects", type: "folder", children: [] },
      0,
    );
    expect(d.kind).toBe("directory");
    expect(d.childrenLoaded).toBe(false);
  });

  it("marks nested children loaded when budget remains", () => {
    const d = mapDirent(
      {
        name: "projects",
        path: "/ws/projects",
        type: "folder",
        children: [{ name: "water", path: "/ws/projects/water", type: "folder", children: [] }],
      },
      1,
    );
    expect(d.childrenLoaded).toBe(true);
    expect(d.children[0]?.childrenLoaded).toBe(false);
  });
});

describe("mergeTreeChildren", () => {
  it("replaces children at the matching path", () => {
    const root: WorkspaceTreeNode = {
      id: "workspace-root",
      name: "ws",
      path: "/ws",
      kind: "directory",
      children: [
        {
          id: "/ws/projects",
          name: "projects",
          path: "/ws/projects",
          kind: "directory",
          children: [],
          sizeBytes: 0,
          updatedAt: "",
          childrenLoaded: false,
        },
      ],
      sizeBytes: 0,
      updatedAt: "",
      childrenLoaded: true,
    };
    const kids: WorkspaceTreeNode[] = [
      {
        id: "/ws/projects/water",
        name: "water",
        path: "/ws/projects/water",
        kind: "directory",
        children: [],
        sizeBytes: 0,
        updatedAt: "",
        childrenLoaded: false,
      },
    ];
    const next = mergeTreeChildren(root, "/ws/projects", kids);
    expect(next.children[0]?.childrenLoaded).toBe(true);
    expect(next.children[0]?.children.map((c) => c.name)).toEqual(["water"]);
  });
});

describe("HttpWorkspaceFs.listdir", () => {
  it("calls /api/workspace/files with relative path and maps dirents", async () => {
    const calls: string[] = [];
    const fetchImpl = (async (input: RequestInfo | URL) => {
      calls.push(String(input));
      return new Response(
        JSON.stringify({
          path: "/home/ws",
          children: [
            { name: "projects", path: "/home/ws/projects", type: "folder", children: [] },
            { name: "workspace.json", path: "/home/ws/workspace.json", type: "file", size: 10 },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;
    const fs = new HttpWorkspaceFs({ root: "/home/ws", fetchImpl });
    const entries = await fs.listdir("", { maxDepth: 1 });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toContain("/api/workspace/files?");
    expect(calls[0]).toContain("max_depth=1");
    expect(entries.map((e) => e.name)).toEqual(["projects", "workspace.json"]);
    expect(entries[0]?.childrenLoaded).toBe(false);
    expect(entries[1]?.kind).toBe("file");

    const root = treeRootFromListing("/home/ws", entries);
    expect(root.id).toBe("workspace-root");
    expect(root.children).toHaveLength(2);
  });
});
