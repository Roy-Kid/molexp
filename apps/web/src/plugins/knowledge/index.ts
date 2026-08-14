import { KnowledgeDocPanel } from "@/app/knowledge/KnowledgeDocPanel";
import { registerRendererContribution } from "@/app/registry";
import { KnowledgeViewer } from "@/app/renderers/KnowledgeViewer";
import type { UiPluginModule } from "@/plugins/types";

const knowledgePlugin: UiPluginModule = {
  id: "knowledge",
  name: "Knowledge",
  description: "Notes and literature browser with Milkdown editing.",
  userToggleable: true,
  register: () => {
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
      Component: KnowledgeViewer,
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
      Component: KnowledgeDocPanel,
    });
  },
};

export default knowledgePlugin;
