import type {
  ContentType,
  FileKind,
  PanelKind,
  RendererKey,
  RendererProps,
  SemanticObjectType,
  Selection,
} from "@/app/types";

export type PanelSlot = "center" | "right";

export interface RendererEntry {
  key: RendererKey;
  title: string;
  panelSlot: PanelSlot;
  Component: (props: RendererProps) => JSX.Element;
}

const registry = new Map<string, RendererEntry>();

export const buildRegistryKey = (key: RendererKey): string => {
  return `${key.objectType}::${key.fileKind}::${key.contentType}::${key.panelKind}`;
};

export const registerRenderer = (entry: RendererEntry): void => {
  const key = buildRegistryKey(entry.key);
  if (registry.has(key)) {
    throw new Error(`Renderer already registered for ${key}`);
  }
  registry.set(key, entry);
};

export const resolveRenderer = (key: RendererKey): RendererEntry => {
  const registryKey = buildRegistryKey(key);
  const entry = registry.get(registryKey);
  if (!entry) {
    throw new Error(`No renderer registered for ${registryKey}`);
  }
  return entry;
};

export interface RenderTarget {
  panelKind: PanelKind;
  contentType: ContentType;
  fileKind: FileKind;
}

export interface RenderPlan {
  center: RenderTarget[];
  right: RenderTarget[];
}

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
