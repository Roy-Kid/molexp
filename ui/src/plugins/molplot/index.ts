import { registerFileTypeContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { MolplotObservablesTab } from "./MolplotObservablesTab";

export { MolplotBarChart } from "./MolplotBarChart";
export { MolplotGanttChart } from "./MolplotGanttChart";
export { MolplotLineChart, type MolplotLineChartHandle } from "./MolplotLineChart";
export { MolplotRawChart } from "./MolplotRawChart";

/**
 * molplot UI plugin — activates when run products look plottable.
 *
 * Science path: molpy writes MolRec with ``observables/`` → molexp stores
 * the record under the run → this plugin matches (tags / marker files) and
 * offers a Plots tab. Core never imports chart code for that path.
 */
const molplotPlugin: UiPluginModule = {
  id: "molplot",
  register: () => {
    registerFileTypeContribution({
      id: "molplot:run-tab",
      objectType: "run",
      value: "plots",
      label: "Plots",
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
