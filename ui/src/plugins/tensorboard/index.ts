import { registerFileTypeContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { TensorBoardTab } from "./TensorBoardTab";

const tensorboardPlugin: UiPluginModule = {
  id: "tensorboard",
  register: () => {
    registerFileTypeContribution({
      id: "tensorboard:run-tab",
      objectType: "run",
      value: "tensorboard",
      label: "TensorBoard",
      priority: 45,
      matcher: {
        // tfevents files always start with this prefix. The wildcard
        // segment is the hostname / suffix appended by tensorboard's
        // EventFileWriter — e.g. ``events.out.tfevents.1700000000.host.0``.
        patterns: ["events.out.tfevents.*", "**/events.out.tfevents.*"],
      },
      Component: TensorBoardTab,
    });
  },
};

export default tensorboardPlugin;
