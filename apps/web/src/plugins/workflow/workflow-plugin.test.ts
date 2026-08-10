/**
 * Workflow UI plugin contract — contribution ids, pluginId stamping, and
 * user-disable filtering. Does not import the real Flowgram-backed viewers
 * (CSS/DOM); stubs stand in for Components.
 */

import { beforeEach, describe, expect, it } from "@rstest/core";
import { registerRendererContribution, tryResolveRenderer } from "@/app/registry";
import {
  resetContributionRuntimeForTests,
  runWithPluginContext,
} from "@/plugins/contribution-runtime";
import {
  isPluginEnabled,
  resetPluginPreferencesForTests,
  setPluginEnabled,
} from "@/plugins/preferences";
import type { UiPluginModule } from "@/plugins/types";

// Import only the module metadata path — re-declare the register body with
// stubs so rstest never loads flowgram CSS.
const workflowMeta: Pick<UiPluginModule, "id" | "name" | "userToggleable"> = {
  id: "workflow",
  name: "Workflow",
  userToggleable: true,
};

const registerWorkflowContributions = (): void => {
  const Stub = (() => null) as never;
  registerRendererContribution({
    id: "workflow:viewer",
    key: {
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Workflow Overview",
    panelSlot: "center",
    priority: 0,
    Component: Stub,
  });
  registerRendererContribution({
    id: "workflow:inspector",
    key: {
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Workflow Inspector",
    panelSlot: "right",
    priority: 0,
    Component: Stub,
  });
  registerRendererContribution({
    id: "workflow:file:workspace-file::json::workflow-graph::viewer",
    key: {
      objectType: "workspace-file",
      fileKind: "json",
      contentType: "workflow-graph",
      panelKind: "viewer",
    },
    title: "Workflow Preview",
    panelSlot: "center",
    priority: 0,
    Component: Stub,
  });
};

beforeEach(() => {
  resetContributionRuntimeForTests();
  resetPluginPreferencesForTests();
});

describe("workflow plugin", () => {
  it("exposes a toggleable workflow catalog entry shape", () => {
    expect(workflowMeta.id).toBe("workflow");
    expect(workflowMeta.name).toBe("Workflow");
    expect(workflowMeta.userToggleable).toBe(true);
  });

  it("registers viewer + inspector under pluginId workflow", () => {
    runWithPluginContext("workflow", () => {
      registerWorkflowContributions();
    });

    const viewer = tryResolveRenderer({
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "viewer",
    });
    const inspector = tryResolveRenderer({
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "inspector",
    });
    const fileViewer = tryResolveRenderer({
      objectType: "workspace-file",
      fileKind: "json",
      contentType: "workflow-graph",
      panelKind: "viewer",
    });

    expect(viewer?.title).toBe("Workflow Overview");
    expect(viewer?.pluginId).toBe("workflow");
    expect(inspector?.title).toBe("Workflow Inspector");
    expect(inspector?.pluginId).toBe("workflow");
    expect(fileViewer?.title).toBe("Workflow Preview");
    expect(fileViewer?.pluginId).toBe("workflow");
  });

  it("disappears from resolve when the plugin is user-disabled", () => {
    runWithPluginContext("workflow", () => {
      registerWorkflowContributions();
    });
    registerRendererContribution({
      id: "test:fallback-run",
      pluginId: "core",
      key: {
        objectType: "run",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "viewer",
      },
      title: "Core Run",
      panelSlot: "center",
      priority: 0,
      Component: (() => null) as never,
    });
    runWithPluginContext("molq", () => {
      registerRendererContribution({
        id: "molq:run-viewer",
        key: {
          objectType: "run",
          fileKind: "json",
          contentType: "metadata",
          panelKind: "viewer",
        },
        title: "Molq Run",
        panelSlot: "center",
        priority: 100,
        Component: (() => null) as never,
      });
    });

    expect(
      tryResolveRenderer({
        objectType: "run",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "viewer",
      })?.title,
    ).toBe("Molq Run");

    setPluginEnabled("molq", false);
    expect(isPluginEnabled("molq")).toBe(false);
    expect(
      tryResolveRenderer({
        objectType: "run",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "viewer",
      })?.title,
    ).toBe("Core Run");

    setPluginEnabled("workflow", false);
    expect(
      tryResolveRenderer({
        objectType: "workflow",
        fileKind: "yaml",
        contentType: "metadata",
        panelKind: "viewer",
      }),
    ).toBeNull();
  });
});
