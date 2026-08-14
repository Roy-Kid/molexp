import { registerRenderer } from "@/app/registry";
import { AgentSessionInspector } from "@/app/renderers/AgentSessionInspector";
import { AgentViewer } from "@/app/renderers/AgentViewer";
import { AssetViewer } from "@/app/renderers/AssetViewer";
import { ExperimentViewer } from "@/app/renderers/ExperimentViewer";
import { ImageViewer } from "@/app/renderers/ImageViewer";
import { MetadataInspector } from "@/app/renderers/MetadataInspector";
import { ProjectViewer } from "@/app/renderers/ProjectViewer";
import { RunViewer } from "@/app/renderers/RunViewer";
import { TaskViewer } from "@/app/renderers/TaskViewer";
// WorkflowViewer / WorkflowInspector / WorkflowFileViewer live in the
// internal `workflow` UI plugin (`@/plugins/workflow`), registered by
// `bootPlugins()` — same pattern as the `editor` panel slot.

export const registerDefaultRenderers = (): void => {
  registerRenderer({
    key: {
      objectType: "project",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Project Overview",
    panelSlot: "center",
    Component: ProjectViewer,
  });

  registerRenderer({
    key: {
      objectType: "experiment",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Experiment Overview",
    panelSlot: "center",
    Component: ExperimentViewer,
  });

  registerRenderer({
    key: {
      objectType: "run",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Run Overview",
    panelSlot: "center",
    Component: RunViewer,
  });

  registerRenderer({
    key: {
      objectType: "asset",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Asset Overview",
    panelSlot: "center",
    Component: AssetViewer,
  });

  // The `panelKind:"editor"` renderer is owned by the internal `editor`
  // plugin (`@/plugins/editor`), registered eagerly in `bootPlugins()`.
  // Workflow center/right + workflow.json preview: `@/plugins/workflow`.
  const workspaceFileKinds = [
    "yaml",
    "json",
    "python",
    "markdown",
    "text",
    "unknown",
    "image",
  ] as const;

  registerRenderer({
    key: {
      objectType: "workspace-file",
      fileKind: "image",
      contentType: "image",
      panelKind: "viewer",
    },
    title: "Image Preview",
    panelSlot: "center",
    Component: ImageViewer,
  });

  registerRenderer({
    key: {
      objectType: "project",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Project Inspector",
    panelSlot: "right",
    Component: MetadataInspector,
  });

  registerRenderer({
    key: {
      objectType: "experiment",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Experiment Inspector",
    panelSlot: "right",
    Component: MetadataInspector,
  });

  registerRenderer({
    key: {
      objectType: "run",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Run Inspector",
    panelSlot: "right",
    Component: MetadataInspector,
  });

  registerRenderer({
    key: {
      objectType: "asset",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Asset Inspector",
    panelSlot: "right",
    Component: MetadataInspector,
  });

  workspaceFileKinds.forEach((fileKind) => {
    registerRenderer({
      key: {
        objectType: "workspace-file",
        fileKind,
        contentType: "metadata",
        panelKind: "inspector",
      },
      title: "File Inspector",
      panelSlot: "right",
      Component: MetadataInspector,
    });
  });

  registerRenderer({
    key: {
      objectType: "agent",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Agent Task",
    panelSlot: "center",
    Component: AgentViewer,
  });

  registerRenderer({
    key: {
      objectType: "agent",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Agent Task Inspector",
    panelSlot: "right",
    Component: AgentSessionInspector,
  });

  registerRenderer({
    key: {
      objectType: "task",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Task Overview",
    panelSlot: "center",
    Component: TaskViewer,
  });

  registerRenderer({
    key: {
      objectType: "task",
      fileKind: "json",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Task Inspector",
    panelSlot: "right",
    Component: TaskViewer,
  });
};
