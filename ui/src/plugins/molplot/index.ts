import type { UiPluginModule } from "@/plugins/types";

export { MolplotBarChart } from "./MolplotBarChart";
export { MolplotGanttChart } from "./MolplotGanttChart";
export { MolplotLineChart, type MolplotLineChartHandle } from "./MolplotLineChart";
export { MolplotRawChart } from "./MolplotRawChart";

/**
 * molexp-side integration of `@molcrafts/molplot` — the plotly-backed
 * charting primitives split out of `@molcrafts/molvis-core`. Mirrors the
 * `molvis` plugin (which integrates the babylon.js 3D viewer): each chart
 * is a thin React wrapper that lazy-imports molplot so plotly lands in an
 * async chunk.
 *
 * The wrappers are consumed directly as components (no file-type or preview
 * contribution to register), so `register()` is intentionally empty. The
 * module is still installed in `bootPlugins()` for symmetry with the other
 * internal plugins and as the home for any future molplot contributions.
 */
const molplotPlugin: UiPluginModule = {
  id: "molplot",
  register: () => {},
};

export default molplotPlugin;
