import { registerRendererContribution } from "@/app/registry";
import { MolqRunInspector } from "@/plugins/molq/MolqRunInspector";
import { MolqRunViewer } from "@/plugins/molq/MolqRunViewer";
import type { UiPluginModule } from "@/plugins/types";

const isMolqRun = (runId: string, runs: Array<{ id: string; executorInfo: Record<string, string> }>) => {
  const run = runs.find((item) => item.id === runId);
  return run?.executorInfo.backend === "molq";
};

const molqPlugin: UiPluginModule = {
  id: "molq",
  register: () => {
    registerRendererContribution({
      id: "molq:run-viewer",
      key: {
        objectType: "run",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "viewer",
      },
      title: "Molq Run Monitor",
      panelSlot: "center",
      priority: 100,
      matches: ({ selection, snapshot }) => isMolqRun(selection.objectId, snapshot.runs),
      Component: MolqRunViewer,
    });

    registerRendererContribution({
      id: "molq:run-inspector",
      key: {
        objectType: "run",
        fileKind: "json",
        contentType: "metadata",
        panelKind: "inspector",
      },
      title: "Molq Run Inspector",
      panelSlot: "right",
      priority: 100,
      matches: ({ selection, snapshot }) => isMolqRun(selection.objectId, snapshot.runs),
      Component: MolqRunInspector,
    });
  },
};

export default molqPlugin;
