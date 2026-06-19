import { registerFileTypeContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { DeltaFChart } from "./DeltaFChart";

/**
 * ΔF plugin — auto-discovers a Phase-1 quantization run by its
 * `phase1_df_report.json` artifact and adds a "ΔF" tab rendering the
 * force-deviation decomposition (vs the fp32 gold standard) as a molplot bar
 * chart. Pure filename-glob discovery, same mechanism as the metrics plugin.
 */
const deltafPlugin: UiPluginModule = {
  id: "deltaf",
  register: () => {
    registerFileTypeContribution({
      id: "deltaf:run-tab",
      objectType: "run",
      value: "deltaf",
      label: "ΔF",
      priority: 48,
      matcher: {
        patterns: ["phase1_df_report.json", "**/phase1_df_report.json"],
      },
      Component: DeltaFChart,
    });
  },
};

export default deltafPlugin;
