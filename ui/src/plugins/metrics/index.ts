import { registerEntityTabContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { RunMetricsTab } from "./RunMetricsTab";

const metricsPlugin: UiPluginModule = {
  id: "metrics",
  register: () => {
    registerEntityTabContribution({
      id: "metrics:run-tab",
      objectType: "run",
      value: "metrics",
      label: "Metrics",
      priority: 50,
      Component: RunMetricsTab,
    });
  },
};

export default metricsPlugin;
