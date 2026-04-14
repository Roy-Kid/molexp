/**
 * Tests for app/registry.ts — renderer registry and key-building functions.
 *
 * Per project convention:
 * - describe('functionName') wraps each exported function
 * - it('...') covers one behaviour per case
 * - registry is module-level state; clear it between tests
 */

import { beforeEach, describe, expect, it } from "@rstest/core";
import type { RendererEntry } from "@/app/registry";
import {
  buildRegistryKey,
  buildRendererKeyFromSelection,
  registerRenderer,
  registerRendererContribution,
  renderPlanByObjectType,
  resolveRenderer,
} from "@/app/registry";
import { resetUiPluginsForTests } from "@/plugins/runtime";
import type { RendererKey, WorkspaceSnapshot } from "@/app/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEntry(key: RendererKey, title = "Test"): RendererEntry {
  return {
    key,
    title,
    panelSlot: "center",
    Component: (() => null) as unknown as RendererEntry["Component"],
  };
}

// ---------------------------------------------------------------------------

beforeEach(() => {
  resetUiPluginsForTests();
});

describe("buildRegistryKey", () => {
  it("joins all four parts with '::'", () => {
    const key: RendererKey = {
      objectType: "project",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    };
    expect(buildRegistryKey(key)).toBe("project::json::metadata::viewer");
  });

  it("produces consistent output for identical inputs", () => {
    const key: RendererKey = {
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "workflow-graph",
      panelKind: "inspector",
    };
    expect(buildRegistryKey(key)).toBe(buildRegistryKey(key));
  });

  it("distinguishes keys that differ only in one field", () => {
    const base: RendererKey = {
      objectType: "run",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    };
    const diffPanel: RendererKey = { ...base, panelKind: "inspector" };
    expect(buildRegistryKey(base)).not.toBe(buildRegistryKey(diffPanel));
  });
});

describe("registerRenderer", () => {
  it("registers without throwing", () => {
    const key: RendererKey = {
      objectType: "asset",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    };
    expect(() => registerRenderer(makeEntry(key))).not.toThrow();
  });

  it("throws when the same key is registered twice", () => {
    const key: RendererKey = {
      objectType: "experiment",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    };
    registerRenderer(makeEntry(key, "First"));
    expect(() => registerRenderer(makeEntry(key, "Second"))).toThrow(/already registered/);
  });
});

describe("resolveRenderer", () => {
  it("returns the registered entry", () => {
    const key: RendererKey = {
      objectType: "run",
      fileKind: "json",
      contentType: "log",
      panelKind: "viewer",
    };
    const entry = makeEntry(key, "LogViewer");
    registerRenderer(entry);
    expect(resolveRenderer(key).title).toBe("LogViewer");
  });

  it("throws for an unregistered key", () => {
    const key: RendererKey = {
      objectType: "asset",
      fileKind: "image",
      contentType: "image",
      panelKind: "inspector",
    };
    expect(() => resolveRenderer(key)).toThrow(/No renderer registered/);
  });

  it("prefers higher-priority contribution when runtime context matches", () => {
    const key: RendererKey = {
      objectType: "run",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    };
    registerRenderer(makeEntry(key, "Default"));
    registerRendererContribution({
      id: "molq:test-viewer",
      key,
      title: "Molq",
      panelSlot: "center",
      priority: 100,
      matches: ({ selection, snapshot }) => {
        const run = snapshot.runs.find((item) => item.id === selection.objectId);
        return run?.executorInfo.backend === "molq";
      },
      Component: (() => null) as unknown as RendererEntry["Component"],
    });

    const snapshot = {
      projects: [],
      experiments: [],
      runs: [
        {
          id: "run-1",
          name: "run-1",
          status: "running",
          summary: "Status: running",
          updatedAt: "2026-01-01T00:00:00Z",
          projectId: "proj-1",
          experimentId: "exp-1",
          executorInfo: { backend: "molq" },
        },
      ],
      assets: [],
      workflows: [],
      agentSessions: [],
      workspaceRoot: null,
      consoleEntries: [],
    } satisfies WorkspaceSnapshot;

    expect(
      resolveRenderer(key, {
        selection: { objectType: "run", objectId: "run-1" },
        snapshot,
        target: { panelKind: "viewer", contentType: "metadata", fileKind: "json" },
      }).title,
    ).toBe("Molq");
  });
});

describe("renderPlanByObjectType", () => {
  const types = ["project", "experiment", "run", "asset", "workflow", "workspace-file"] as const;

  it("has an entry for every SemanticObjectType", () => {
    for (const t of types) {
      expect(renderPlanByObjectType).toHaveProperty(t);
    }
  });

  it("each entry has non-empty center and right arrays", () => {
    for (const t of types) {
      expect(renderPlanByObjectType[t].center.length).toBeGreaterThan(0);
      expect(renderPlanByObjectType[t].right.length).toBeGreaterThan(0);
    }
  });

  it("workflow uses yaml fileKind in center", () => {
    const [centerTarget] = renderPlanByObjectType.workflow.center;
    expect(centerTarget.fileKind).toBe("yaml");
  });

  it("workspace-file center uses editor panelKind", () => {
    const [centerTarget] = renderPlanByObjectType["workspace-file"].center;
    expect(centerTarget.panelKind).toBe("editor");
  });
});

describe("buildRendererKeyFromSelection", () => {
  it("uses target fileKind for non-workspace-file selection", () => {
    const selection = { objectType: "project" as const, objectId: "proj-1" };
    const target = {
      panelKind: "viewer" as const,
      contentType: "metadata" as const,
      fileKind: "json" as const,
    };
    const result = buildRendererKeyFromSelection(selection, target);
    expect(result.fileKind).toBe("json");
    expect(result.objectType).toBe("project");
  });

  it("returns workflow-graph viewer key for workspace-file editor targeting workflow.json", () => {
    const selection = {
      objectType: "workspace-file" as const,
      objectId: "/ws/workflow.json",
      filePath: "/ws/workflow.json",
      fileKind: "json" as const,
    };
    const target = {
      panelKind: "editor" as const,
      contentType: "text" as const,
      fileKind: "text" as const,
    };
    const result = buildRendererKeyFromSelection(selection, target);
    expect(result.contentType).toBe("workflow-graph");
    expect(result.panelKind).toBe("viewer");
    expect(result.fileKind).toBe("json");
  });

  it("returns image viewer key for workspace-file editor with image fileKind", () => {
    const selection = {
      objectType: "workspace-file" as const,
      objectId: "/ws/photo.png",
      filePath: "/ws/photo.png",
      fileKind: "image" as const,
    };
    const target = {
      panelKind: "editor" as const,
      contentType: "text" as const,
      fileKind: "text" as const,
    };
    const result = buildRendererKeyFromSelection(selection, target);
    expect(result.contentType).toBe("image");
    expect(result.panelKind).toBe("viewer");
    expect(result.fileKind).toBe("image");
  });

  it("returns regular key for workspace-file non-editor target", () => {
    const selection = {
      objectType: "workspace-file" as const,
      objectId: "/ws/notes.md",
      filePath: "/ws/notes.md",
      fileKind: "markdown" as const,
    };
    const target = {
      panelKind: "inspector" as const,
      contentType: "metadata" as const,
      fileKind: "text" as const,
    };
    const result = buildRendererKeyFromSelection(selection, target);
    expect(result.objectType).toBe("workspace-file");
    expect(result.panelKind).toBe("inspector");
    expect(result.fileKind).toBe("markdown");
  });

  it("uses selection fileKind (not target fileKind) for workspace-file", () => {
    const selection = {
      objectType: "workspace-file" as const,
      objectId: "/ws/data.yaml",
      filePath: "/ws/data.yaml",
      fileKind: "yaml" as const,
    };
    const target = {
      panelKind: "inspector" as const,
      contentType: "metadata" as const,
      fileKind: "text" as const,
    };
    const result = buildRendererKeyFromSelection(selection, target);
    expect(result.fileKind).toBe("yaml");
  });
});
