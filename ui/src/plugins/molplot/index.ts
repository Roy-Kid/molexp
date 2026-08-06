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
 * - MolRec metrics **JSONL buffer** (``metrics/metrics.jsonl``) → Metrics tab.
 *   Host ``metrics/index.json`` is a series cache only — never matched here.
 * - Vega-Lite artifacts → Plots tab. Full MolRec ``observables/`` Zarr arrays
 *   still need a server/molrs reader (not a browser Zarr open yet).
 *
 * Core never hard-wires chart tabs; matching is contribution-driven.
 */
const isMetricsJsonl = (file: { name: string; relPath: string }): boolean => {
  const path = `${file.relPath}`.toLowerCase().replace(/\\/g, "/");
  const name = file.name.toLowerCase();
  // Canonical buffer path only — not index.json, not random *.jsonl.
  return (
    path.endsWith("metrics/metrics.jsonl") ||
    path === "metrics.jsonl" ||
    (name === "metrics.jsonl" && path.includes("/metrics/"))
  );
};

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
        patterns: ["metrics/metrics.jsonl", "**/metrics/metrics.jsonl"],
        matches: isMetricsJsonl,
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
          "*.vl.json",
          "**/*.vl.json",
        ],
        matches: (file) => {
          const path = `${file.relPath}`.toLowerCase().replace(/\\/g, "/");
          const name = file.name.toLowerCase();
          // Explicit plot artifacts only — no free-text "molrec" / "plot" heuristics.
          if (name.endsWith(".vl.json") || path.endsWith(".vl.json")) return true;
          if (name === ".molexp-artifact.json" || path.endsWith("/.molexp-artifact.json")) {
            return true;
          }
          return false;
        },
      },
      Component: MolplotObservablesTab,
    });
  },
};

export default molplotPlugin;
