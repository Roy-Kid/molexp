/**
 * Retired product surface.
 *
 * Charts and scientific series are owned by the **molplot** plugin only.
 * This module remains only so old imports of `smoothEma` can resolve; it
 * no longer registers any run tabs.
 */

import type { UiPluginModule } from "@/plugins/types";

const metricsPlugin: UiPluginModule = {
  id: "metrics",
  name: "Metrics (retired)",
  description: "Retired — charts live under MolPlot.",
  userToggleable: false,
  register: () => {
    // intentionally empty — no Metrics / Telemetry tab
  },
};

export default metricsPlugin;
export { smoothEma } from "./smoothing";
