import type React from "react";
import type {
  ContentType,
  FileKind,
  PanelKind,
  RendererKey,
  RendererProps,
  Selection,
  WorkspaceSnapshot,
} from "@/app/types";

export type PanelSlot = "center" | "right";

export interface RendererEntry {
  key: RendererKey;
  title: string;
  panelSlot: PanelSlot;
  Component: React.ComponentType<RendererProps>;
}

export interface RenderTarget {
  panelKind: PanelKind;
  contentType: ContentType;
  fileKind: FileKind;
}

export interface RendererResolutionContext {
  key: RendererKey;
  selection: Selection;
  snapshot: WorkspaceSnapshot;
  target: RenderTarget;
}

export interface RendererContribution extends RendererEntry {
  id: string;
  priority?: number;
  matches?: (context: RendererResolutionContext) => boolean;
}

export interface FilePreviewContentProps {
  content: string;
  name: string;
  path: string;
  folderId: string;
}

export interface FilePreviewPlugin {
  id: string;
  name: string;
  extensions: string[];
  priority?: number;
  canHandle?: (props: { name: string; path: string }) => boolean;
  Component: React.ComponentType<FilePreviewContentProps>;
}

export interface PluginManifest {
  id: string;
  title: string;
  description: string;
  uiModule?: string | null;
  capabilities: string[];
  metadata: Record<string, unknown>;
}

export interface UiPluginModule {
  id: string;
  register: () => void | Promise<void>;
}

export const buildRendererRegistryKey = (key: RendererKey): string => {
  return `${key.objectType}::${key.fileKind}::${key.contentType}::${key.panelKind}`;
};
