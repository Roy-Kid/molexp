import type { BarChartConfig } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { MolplotBarChart } from "@/plugins/molplot";
import type { DiscoveredFile } from "@/plugins/types";

type DeltaFTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

interface DeltaFStats {
  comp_rms_mev_A: number;
  atom_rms_mev_A: number;
  max_mev_A: number;
  pct_of_Frms: number;
}

interface DeltaFReport {
  gold_standard?: string;
  eval_configs?: number;
  dataset_force_rms_mev_A?: number;
  deltaF: Record<string, DeltaFStats>;
}

// Stable display order + human labels for the ΔF decomposition keys produced
// by eval_df.compute_deltaF (phase1).
// Bars grouped by effect type, coloured with the SciencePlots "science" cycle
// (MIT-licensed style; these are plain hex config values). Short tick labels
// (long ones need `automargin`, which oscillates margins on zoom → jitter); the
// full description lives in the hover. Each tuple = [reportKey, tick, hover].
type Member = readonly [string, string, string];
const GROUPS: ReadonlyArray<{
  id: string;
  label: string;
  color: string;
  members: ReadonlyArray<Member>;
}> = [
  {
    id: "inference",
    label: "inference (bf16 forward)",
    color: "#0C5DA5", // science blue
    members: [
      ["infer_effect_on_fp32w", "infer·fp32w", "inference bf16, fp32 weights"],
      ["infer_effect_on_bf16w", "infer·bf16w", "inference bf16, bf16 weights"],
    ],
  },
  {
    id: "training",
    label: "training precision",
    color: "#FF9500", // science amber
    members: [
      ["train_effect_fp64", "train·fp64", "fp64 training vs fp32"],
      ["train_effect_bf16", "train·bf16", "bf16 training vs fp32"],
    ],
  },
  {
    id: "combined",
    label: "combined / reference",
    color: "#845B97", // science purple
    members: [
      ["combined_e2e_bf16", "e2e·bf16", "combined end-to-end bf16"],
      ["fp64_native_vs_gold", "fp64·native", "fp64 native vs gold"],
    ],
  },
];

/**
 * Auto-discovered run tab for the Phase-1 ΔF report (`phase1_df_report.json`).
 * Renders the per-component force-deviation decomposition (vs the fp32 gold
 * standard) as a molplot bar chart; hover shows the % of the dataset force RMS.
 */
export const DeltaFChart = ({
  selection,
  snapshot,
  discoveredFiles,
}: DeltaFTabProps): JSX.Element => {
  const run = snapshot.runs.find((item) => item.id === selection.objectId) ?? null;
  // Depend on the stable string ids, NOT the `run` object: snapshot/SSE updates
  // hand back a fresh `run` reference every parent render, which would re-fire
  // the fetch → setReport → new config → MolplotBarChart dispose+recreate
  // (visible flicker). Strings are value-stable, so this fetches exactly once.
  const projectId = run?.projectId ?? null;
  const experimentId = run?.experimentId ?? null;
  const runId = run?.id ?? null;
  const relPath = discoveredFiles?.[0]?.relPath ?? "phase1_df_report.json";

  const [report, setReport] = useState<DeltaFReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId || !experimentId || !runId) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await workspaceApi.getRunFileText(projectId, experimentId, runId, relPath);
        if (cancelled) {
          return;
        }
        setReport(JSON.parse(res.content) as DeltaFReport);
        setError(null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "failed to load ΔF report");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, experimentId, runId, relPath]);

  const config = useMemo<BarChartConfig | null>(() => {
    if (!report?.deltaF) {
      return null;
    }
    const series = GROUPS.map((group) => ({
      id: group.id,
      label: group.label,
      color: group.color,
      points: group.members
        .filter(([key]) => report.deltaF[key] != null)
        .map(([key, label, desc]) => {
          const stat = report.deltaF[key];
          return {
            x: label,
            y: stat.comp_rms_mev_A,
            // NOTE: deliberately no `text` — molplot renders point.text ON the
            // bar with auto-shrunk font (unreadable on small bars). Values are
            // read off the (enlarged) y-axis; full detail is in the hover via
            // customdata.
            customdata: `${desc}: ${stat.comp_rms_mev_A.toFixed(1)} meV/Å · ${stat.pct_of_Frms.toFixed(2)}% of F_rms`,
          };
        }),
      hovertemplate: "%{customdata}<extra></extra>",
    })).filter((s) => s.points.length > 0);
    return {
      series,
      // overlay (not group): categories are disjoint across series, so each bar
      // centres at its own tick instead of being offset within a cluster.
      mode: "overlay",
      xAxis: { label: "", tickfont: { size: 13 } },
      yAxis: { label: "ΔF (meV/Å)", rangemode: "tozero", tickfont: { size: 13 } },
      showLegend: true,
    };
  }, [report]);

  if (!runId) {
    return <div className="p-4 text-sm text-muted-foreground">No run selected.</div>;
  }
  if (error) {
    return <div className="p-4 text-sm text-red-500">ΔF report: {error}</div>;
  }
  if (!config) {
    return <div className="p-4 text-sm text-muted-foreground">Loading ΔF report…</div>;
  }
  return (
    <div className="flex flex-col gap-2 p-4">
      <div className="text-sm text-muted-foreground">
        Gold standard: {report?.gold_standard} · {report?.eval_configs} configs · F_rms ={" "}
        {report?.dataset_force_rms_mev_A} meV/Å
      </div>
      {/* Fixed height (not flex-1/100%) so Plotly's autosize can't feed back
          into a resizing flex parent — the same stable pattern the metrics tab
          uses. */}
      <MolplotBarChart config={config} style={{ width: "100%", height: "460px" }} />
    </div>
  );
};
