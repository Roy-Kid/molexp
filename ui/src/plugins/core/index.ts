import { MarkdownPreview } from "@/components/previews/MarkdownPreview";
import { WorkflowPreview } from "@/components/previews/WorkflowPreview";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import type { UiPluginModule } from "@/plugins/types";
import { registerDefaultRenderers } from "@/app/renderers/registerRenderers";

const corePlugin: UiPluginModule = {
  id: "core",
  register: () => {
    registerDefaultRenderers();

    filePreviewPluginRegistry.register({
      id: "core:markdown-preview",
      name: "Markdown Preview",
      extensions: [".md", ".markdown"],
      priority: 20,
      Component: MarkdownPreview,
    });

    filePreviewPluginRegistry.register({
      id: "core:workflow-preview",
      name: "Workflow Preview",
      extensions: [],
      priority: 30,
      canHandle: ({ name, path }) => {
        const candidate = `${name} ${path}`.toLowerCase();
        return candidate.includes("workflow.");
      },
      Component: WorkflowPreview,
    });
  },
};

export default corePlugin;
