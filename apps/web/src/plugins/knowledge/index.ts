import { KnowledgeDocPanel } from "@/plugins/knowledge/KnowledgeDocPanel";
import { registerRendererContribution } from "@/app/registry";
import { KnowledgeViewer } from "@/plugins/knowledge/KnowledgeViewer";
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

export { DocTree } from "./DocTree";
export { KnowledgeBacklinksCard } from "./KnowledgeBacklinksCard";
export { KnowledgeDocPanel } from "./KnowledgeDocPanel";
export { KnowledgeViewer } from "./KnowledgeViewer";
export { NoteEditor } from "./NoteEditor";
export default knowledgePlugin;
