/**
 * Internal `workflow` UI plugin — owns the workflow entity center/right
 * surfaces and the workspace-file `workflow.json` graph preview.
 *
 * Previously registered by `core` via `registerDefaultRenderers`. As a
 * first-class plugin it can be user-toggled in Settings, and host panels
 * skip it when there is no workflow data (see ExperimentViewer).
 */

import { buildRegistryKey, registerRendererContribution } from "@/app/registry";
import { WorkflowFileViewer } from "@/app/renderers/WorkflowFileViewer";
import { WorkflowInspector } from "@/app/renderers/WorkflowInspector";
import { WorkflowViewer } from "@/app/renderers/WorkflowViewer";
import type { UiPluginModule } from "@/plugins/types";

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

export default workflowPlugin;
