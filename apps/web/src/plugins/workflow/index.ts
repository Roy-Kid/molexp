import "./side-effects";

/**
 * Internal `workflow` UI plugin — owns the workflow entity center/right
 * surfaces and the workspace-file `workflow.json` graph preview.
 *
 * Previously registered by `core` via `registerDefaultRenderers`. As a
 * first-class plugin it can be user-toggled in Settings, and host panels
 * skip it when there is no workflow data (see ExperimentViewer).
 */

import { buildRegistryKey, registerRendererContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { WorkflowFileViewer } from "@/plugins/workflow/WorkflowFileViewer";
import { WorkflowInspector } from "@/plugins/workflow/WorkflowInspector";
import { WorkflowViewer } from "@/plugins/workflow/WorkflowViewer";

const workflowPlugin: UiPluginModule = {
  id: "workflow",
  name: "Workflow",
  description: "Workflow graph viewer, source tab, and right-rail inspector.",
  userToggleable: true,
  register: () => {
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
      Component: WorkflowViewer,
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
      Component: WorkflowInspector,
    });

    const fileKey = {
      objectType: "workspace-file" as const,
      fileKind: "json" as const,
      contentType: "workflow-graph" as const,
      panelKind: "viewer" as const,
    };
    registerRendererContribution({
      id: `workflow:file:${buildRegistryKey(fileKey)}`,
      key: fileKey,
      title: "Workflow Preview",
      panelSlot: "center",
      priority: 0,
      Component: WorkflowFileViewer,
    });
  },
};

export { FlowgramCanvas, type FlowgramCanvasProps } from "./flowgram-canvas";
export {
  buildFlowgramDocument,
  buildWorkflowDocument,
  type FlowgramDocument,
  parseTaskGraphIr,
} from "./flowgram-document";
export type { TaskGraphJson, TaskNodeJson } from "./task-graph-ir";
export default workflowPlugin;
