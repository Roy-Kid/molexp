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

const knowledgeMeta: Pick<UiPluginModule, "id" | "name" | "userToggleable"> = {
  id: "knowledge",
  name: "Knowledge",
  userToggleable: true,
};

const registerKnowledgeContributions = (): void => {
  const Stub = (() => null) as never;
  registerRendererContribution({
    id: "knowledge:viewer",
    key: {
      objectType: "knowledge",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Knowledge",
    panelSlot: "center",
    priority: 0,
    Component: Stub,
  });
  registerRendererContribution({
    id: "knowledge:inspector",
    key: {
      objectType: "knowledge",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Document",
    panelSlot: "right",
    priority: 0,
    Component: Stub,
  });
};

beforeEach(() => {
  resetContributionRuntimeForTests();
  resetPluginPreferencesForTests();
});

describe("knowledge plugin", () => {
  it("exposes a toggleable knowledge catalog entry shape", () => {
    expect(knowledgeMeta.id).toBe("knowledge");
    expect(knowledgeMeta.name).toBe("Knowledge");
    expect(knowledgeMeta.userToggleable).toBe(true);
  });

  it("registers viewer + inspector under pluginId knowledge", () => {
    const result = runWithPluginContext("knowledge", () => {
      registerKnowledgeContributions();
    });
    expect(result).toBeUndefined();

    const viewer = tryResolveRenderer({
      objectType: "knowledge",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    });
    const inspector = tryResolveRenderer({
      objectType: "knowledge",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    });
    expect(viewer?.title).toBe("Knowledge");
    expect(viewer?.pluginId).toBe("knowledge");
    expect(inspector?.title).toBe("Document");
    expect(inspector?.pluginId).toBe("knowledge");
  });

  it("disappears from resolve when the plugin is user-disabled", () => {
    runWithPluginContext("knowledge", () => {
      registerKnowledgeContributions();
    });
    expect(isPluginEnabled("knowledge")).toBe(true);
    setPluginEnabled("knowledge", false);
    expect(
      tryResolveRenderer({
        objectType: "knowledge",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "viewer",
      }),
    ).toBeNull();
    expect(
      tryResolveRenderer({
        objectType: "knowledge",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "inspector",
      }),
    ).toBeNull();
  });
});
