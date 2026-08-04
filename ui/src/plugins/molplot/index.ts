import { registerFileTypeContribution } from "@/app/registry";
import { RunMetricsTab } from "@/plugins/metrics/RunMetricsTab";
import type { UiPluginModule } from "@/plugins/types";
import { MolplotObservablesTab } from "./MolplotObservablesTab";

export { MolplotBarChart } from "./MolplotBarChart";
export { MolplotGanttChart } from "./MolplotGanttChart";
export { MolplotLineChart, type MolplotLineChartHandle } from "./MolplotLineChart";
export { MolplotRawChart } from "./MolplotRawChart";

/**
 * molplot UI plugin — activates when run products look plottable.
 *
 * Paths:
 * - MolRec metrics JSONL (``metrics/metrics.jsonl``) → run Metrics tab
 *   (charts via molplot; API reads the same stream molexp/molnex write).
 * - observables / Vega-Lite artifacts → Plots tab.
 *
 * Core never hard-wires chart tabs; matching is contribution-driven.
 */
const molplotPlugin: UiPluginModule = {
  id: "molplot",
  register: () => {
    registerFileTypeContribution({
      id: "molplot:run-metrics",
      objectType: "run",
      value: "metrics",
      label: "Metrics",
      priority: 40,
      matcher: {
        patterns: ["metrics/metrics.jsonl", "**/metrics/metrics.jsonl", "**/metrics.jsonl"],
        matches: (file) => {
          const blob = `${file.name} ${file.relPath}`.toLowerCase().replace(/\\/g, "/");
          return (
            blob.endsWith("metrics/metrics.jsonl") ||
            blob.endsWith("/metrics.jsonl") ||
            blob === "metrics.jsonl"
          );
        },
      },
      Component: RunMetricsTab,
    });
    registerFileTypeContribution({
      id: "molplot:run-tab",
      objectType: "run",
      value: "plots",
      label: "MolPlot",
      priority: 45,
      matcher: {
        patterns: [
          "**/.molexp-artifact.json",
          "**/observables/**",
          "*.vl.json",
          "**/*.vl.json",
          "**/plot*.json",
        ],
        matches: (file) => {
          const blob = `${file.name} ${file.relPath}`.toLowerCase();
          return (
            blob.includes("molrec") ||
            blob.includes("observable") ||
            blob.endsWith(".vl.json") ||
            blob.includes("plot")
          );
        },
      },
      Component: MolplotObservablesTab,
    });
  },
};

export default molplotPlugin;
