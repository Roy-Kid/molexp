import type { RendererKey, Selection, SemanticObjectType } from "@/app/types";
import {
  registerRendererContribution as addRendererContribution,
  resolveRendererContribution,
} from "@/plugins/contribution-runtime";
import type {
  RendererContribution,
  RendererEntry,
  RendererResolutionContext,
  RenderTarget,
} from "@/plugins/types";
import { buildRendererRegistryKey } from "@/plugins/types";

export type { PanelSlot, RendererContribution, RendererEntry, RenderTarget } from "@/plugins/types";

export interface RenderPlan {
  center: RenderTarget[];
  right: RenderTarget[];
}

export const buildRegistryKey = buildRendererRegistryKey;

export const registerRenderer = (entry: RendererEntry): void => {
  const key = buildRegistryKey(entry.key);
  addRendererContribution({
    id: `exact:${key}`,
    ...entry,
  });
};

export const registerRendererContribution = (entry: RendererContribution): void => {
  addRendererContribution(entry);
};

export const resolveRenderer = (
  key: RendererKey,
  context?: Omit<RendererResolutionContext, "key">,
): RendererEntry => {
  const registryKey = buildRegistryKey(key);
  const entry = resolveRendererContribution(key, context);
  if (!entry) {
    throw new Error(`No renderer registered for ${registryKey}`);
  }
  return entry;
};

export const renderPlanByObjectType: Record<SemanticObjectType, RenderPlan> = {
  project: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "json" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "json" }],
  },
  experiment: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "json" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "json" }],
  },
  run: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "json" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "json" }],
  },
  asset: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "json" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "json" }],
  },
  workflow: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "yaml" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "yaml" }],
  },
  "workspace-file": {
    center: [{ panelKind: "editor", contentType: "text", fileKind: "text" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "text" }],
  },
  agent: {
    center: [{ panelKind: "viewer", contentType: "metadata", fileKind: "json" }],
    right: [{ panelKind: "inspector", contentType: "metadata", fileKind: "json" }],
  },
};

export const buildRendererKeyFromSelection = (
  selection: Selection,
  target: RenderTarget,
): RendererKey => {
  const fileKind = selection.objectType === "workspace-file" ? selection.fileKind : target.fileKind;

  if (selection.objectType === "workspace-file" && target.panelKind === "editor") {
    const filePath = selection.objectId.toLowerCase();
    if (filePath.endsWith("workflow.json")) {
      return {
        objectType: "workspace-file",
        fileKind: "json",
        contentType: "workflow-graph",
        panelKind: "viewer",
      };
    }

    if (fileKind === "image") {
      return {
        objectType: "workspace-file",
        fileKind,
        contentType: "image",
        panelKind: "viewer",
      };
    }
  }

  return {
    objectType: selection.objectType,
    fileKind,
    contentType: target.contentType,
    panelKind: target.panelKind,
  };
};
