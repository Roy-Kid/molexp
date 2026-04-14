import { registerRenderer } from "@/app/registry";
import { AgentViewer } from "@/app/renderers/AgentViewer";
import { AssetViewer } from "@/app/renderers/AssetViewer";
import { ExperimentViewer } from "@/app/renderers/ExperimentViewer";
import { ImageViewer } from "@/app/renderers/ImageViewer";
import { MetadataInspector } from "@/app/renderers/MetadataInspector";
import { ProjectViewer } from "@/app/renderers/ProjectViewer";
import { RunViewer } from "@/app/renderers/RunViewer";
import { TextEditor } from "@/app/renderers/TextEditor";
import { WorkflowFileViewer } from "@/app/renderers/WorkflowFileViewer";
import { WorkflowInspector } from "@/app/renderers/WorkflowInspector";
import { WorkflowViewer } from "@/app/renderers/WorkflowViewer";

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

  registerRenderer({
    key: {
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "viewer",
    },
    title: "Workflow Overview",
    panelSlot: "center",
    Component: WorkflowViewer,
  });

  const workspaceFileKinds = [
    "yaml",
    "json",
    "python",
    "markdown",
    "text",
    "unknown",
    "image",
  ] as const;
  const editorFileKinds = ["yaml", "json", "python", "markdown", "text", "unknown"] as const;
  editorFileKinds.forEach((fileKind) => {
    registerRenderer({
      key: {
        objectType: "workspace-file",
        fileKind,
        contentType: "text",
        panelKind: "editor",
      },
      title: "Text Editor",
      panelSlot: "center",
      Component: TextEditor,
    });
  });

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
      objectType: "workspace-file",
      fileKind: "json",
      contentType: "workflow-graph",
      panelKind: "viewer",
    },
    title: "Workflow Preview",
    panelSlot: "center",
    Component: WorkflowFileViewer,
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

  registerRenderer({
    key: {
      objectType: "workflow",
      fileKind: "yaml",
      contentType: "metadata",
      panelKind: "inspector",
    },
    title: "Workflow Inspector",
    panelSlot: "right",
    Component: WorkflowInspector,
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
    title: "Agent Session",
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
    title: "Agent Inspector",
    panelSlot: "right",
    Component: MetadataInspector,
  });
};
