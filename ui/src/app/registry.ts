import type { RendererKey, Selection, SemanticObjectType } from "@/app/types";
import {
  registerEntityTabContribution as addEntityTabContribution,
  registerFileTypeContribution as addFileTypeContribution,
  registerRendererContribution as addRendererContribution,
  listEntityTabContributions,
  listFileTypeContributions,
  resolveRendererContribution,
  unregisterFileTypeContribution,
} from "@/plugins/contribution-runtime";
import type {
  EntityTabContribution,
  FileTypeContribution,
  RendererContribution,
  RendererEntry,
  RendererResolutionContext,
  RenderTarget,
} from "@/plugins/types";
import { buildRendererRegistryKey } from "@/plugins/types";

export type {
  DiscoveredFile,
  EntityTabContribution,
  FileMatchContext,
  FileTypeContribution,
  FileTypeMatcher,
  PanelSlot,
  RendererContribution,
  RendererEntry,
  RenderTarget,
} from "@/plugins/types";

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

export const registerEntityTabContribution = (entry: EntityTabContribution): void => {
  addEntityTabContribution(entry);
};

export const listEntityTabs = (
  objectType: EntityTabContribution["objectType"],
): EntityTabContribution[] => {
  return listEntityTabContributions(objectType);
};

export const registerFileTypeContribution = (entry: FileTypeContribution): void => {
  addFileTypeContribution(entry);
};

export const unregisterFileType = (contributionId: string): boolean => {
  return unregisterFileTypeContribution(contributionId);
};

export const listFileTypes = (
  objectType: FileTypeContribution["objectType"],
): FileTypeContribution[] => {
  return listFileTypeContributions(objectType);
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
    const filePath = (selection.objectId ?? "").toLowerCase();
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
