import { registerFileTypeContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { RunMetricsTab } from "./RunMetricsTab";

const metricsPlugin: UiPluginModule = {
  id: "metrics",
  register: () => {
    registerFileTypeContribution({
      id: "metrics:run-tab",
      objectType: "run",
      value: "metrics",
      label: "Metrics",
      priority: 50,
      matcher: {
        patterns: ["metrics.jsonl", "metrics/*.jsonl", "**/metrics.jsonl"],
      },
      Component: RunMetricsTab,
    });
  },
};

export default metricsPlugin;
