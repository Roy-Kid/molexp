import {
  registerExecutionColumn,
  registerExecutionDetail,
  registerRendererContribution,
} from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { MolqExecutionColumn } from "./MolqExecutionColumn";
import { MolqExecutionDetail } from "./MolqExecutionDetail";
import { MolqRunInspector } from "./MolqRunInspector";
import { MolqRunViewer } from "./MolqRunViewer";

const isMolqRun = (
  runId: string,
  runs: Array<{ id: string; executorInfo: Record<string, string> }>,
) => {
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

    // Workspace Runs page contributions: cluster + scheduler job id columns,
    // plus a detail section that exposes the full molq metadata block.
    registerExecutionColumn({
      id: "molq:column:cluster",
      backend: "molq",
      columnId: "cluster",
      header: "Cluster",
      priority: 100,
      Cell: MolqExecutionColumn.Cluster,
    });
    registerExecutionColumn({
      id: "molq:column:scheduler-job",
      backend: "molq",
      columnId: "scheduler-job",
      header: "Scheduler Job",
      priority: 90,
      Cell: MolqExecutionColumn.SchedulerJob,
    });

    registerExecutionDetail({
      id: "molq:detail:submission",
      backend: "molq",
      title: "Molq submission",
      priority: 100,
      Component: MolqExecutionDetail,
    });
  },
};

export default molqPlugin;
