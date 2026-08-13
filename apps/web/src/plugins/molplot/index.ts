import { registerFileTypeContribution } from "@/app/registry";
import { RunMetricsTab } from "@/plugins/metrics/RunMetricsTab";
import type { UiPluginModule } from "@/plugins/types";
import { MolplotObservablesTab } from "./MolplotObservablesTab";

export { MolplotBarChart } from "./MolplotBarChart";
export { MolplotGanttChart } from "./MolplotGanttChart";
export { MolplotLineChart, type MolplotLineChartHandle } from "./MolplotLineChart";
export { MolplotRawChart } from "./MolplotRawChart";

/**
 * molplot UI plugin — activates purely by filename suffixes (no heuristics).
 *
 * Contract (see molexp.workspace.mlp_names):
 * - ``*.mlp.jsonl`` — live metrics WAL → Metrics tab
 * - ``*.mlp.zarr`` / ``…/*.mlp.zarr/zarr.json`` — dense Zarr SoT → Metrics tab
 * - ``*.mlp.vl.json`` — Vega-Lite plot artifact → MolPlot tab
 * - ``*.mlp.index.json`` is a host cache only — never matched
 */
const isMlpMetricsSurface = (file: { name: string; relPath: string }): boolean => {
  const path = `${file.relPath}`.toLowerCase().replace(/\\/g, "/");
  const name = file.name.toLowerCase();
  if (name.endsWith(".mlp.jsonl") || path.endsWith(".mlp.jsonl")) return true;
  if (name.endsWith(".mlp.zarr") || path.endsWith(".mlp.zarr")) return true;
  if (path.includes(".mlp.zarr/") || (name === "zarr.json" && path.includes(".mlp.zarr"))) {
    return true;
  }
  return false;
};

const isMlpPlotSurface = (file: { name: string; relPath: string }): boolean => {
  const path = `${file.relPath}`.toLowerCase().replace(/\\/g, "/");
  const name = file.name.toLowerCase();
  return name.endsWith(".mlp.vl.json") || path.endsWith(".mlp.vl.json");
};

const molplotPlugin: UiPluginModule = {
  id: "molplot",
  name: "MolPlot",
  description: "Metrics and plot tabs when a run has *.mlp.jsonl / *.mlp.zarr / *.mlp.vl.json.",
  userToggleable: true,
  register: () => {
    registerFileTypeContribution({
      id: "molplot:run-metrics",
      objectType: "run",
      value: "metrics",
      label: "Metrics",
      priority: 40,
      matcher: {
        patterns: ["**/*.mlp.jsonl", "**/*.mlp.zarr", "**/*.mlp.zarr/zarr.json"],
        matches: isMlpMetricsSurface,
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
        patterns: ["**/*.mlp.vl.json", "*.mlp.vl.json"],
        matches: isMlpPlotSurface,
      },
      Component: MolplotObservablesTab,
    });
  },
};

export default molplotPlugin;
