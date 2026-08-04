/**
 * molplot plugin tab — data-driven when MolRec / plot artifacts are present.
 *
 * Vega-Lite artifacts render directly through MolPlot. MolRec directories
 * remain visible in the file rail and explain when a browser reader is needed.
 */

import type { VegaLiteSpec } from "@molcrafts/molplot";
import { BarChart3, FileText } from "lucide-react";
import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { WorkbenchAction } from "@/components/workbench";
import type { DiscoveredFile } from "@/plugins/types";
import { MolplotRawChart } from "./MolplotRawChart";

export const MolplotObservablesTab = ({
  selection,
  snapshot,
  discoveredFiles,
}: RendererProps & { discoveredFiles: DiscoveredFile[] }): JSX.Element => {
  const [selected, setSelected] = useState(discoveredFiles[0]?.relPath ?? "");
  const [spec, setSpec] = useState<VegaLiteSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = useMemo(
    () => snapshot.runs.find((item) => item.id === selection.objectId) ?? null,
    [selection.objectId, snapshot.runs],
  );

  const file = discoveredFiles.find((f) => f.relPath === selected) ?? discoveredFiles[0];

  useEffect(() => {
    if (!file && discoveredFiles[0]) setSelected(discoveredFiles[0].relPath);
  }, [file, discoveredFiles]);

  useEffect(() => {
    if (!file || !run) return;
    let cancelled = false;
    setSpec(null);
    setError(null);
    setLoading(false);

    if (!file.name.toLowerCase().endsWith(".json")) {
      setError(
        "This MolRec observable needs a browser reader. Add a Vega-Lite .vl.json artifact to render it directly in MolPlot.",
      );
      return;
    }

    setLoading(true);
    workspaceApi
      .getRunFileText(run.projectId, run.experimentId, run.id, file.relPath)
      .then((response) => {
        if (cancelled) return;
        const parsed: unknown = JSON.parse(response.content);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("The plot artifact is not a Vega-Lite object.");
        }
        setSpec(parsed as VegaLiteSpec);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Failed to load plot artifact");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [file, run]);

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-56 flex-none flex-col bg-surface/45">
        <div className="flex h-control-comfortable items-center gap-2 px-3">
          <BarChart3 className="size-4 text-accent" aria-hidden />
          <span className="text-label font-medium text-foreground">MolPlot</span>
          <span className="ml-auto font-mono text-micro text-muted-foreground">
            {discoveredFiles.length}
          </span>
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-auto p-2">
          {discoveredFiles.map((candidate) => (
            <WorkbenchAction
              kind="ghost"
              size="content"
              key={candidate.relPath}
              type="button"
              onClick={() => setSelected(candidate.relPath)}
              className={`flex w-full items-center gap-2 px-2 py-2 text-left font-mono text-micro transition-colors ${
                file?.relPath === candidate.relPath
                  ? "bg-accent-muted text-accent-muted-foreground"
                  : "text-muted-foreground hover:bg-interactive hover:text-foreground"
              }`}
            >
              <FileText className="size-3.5 flex-none" aria-hidden />
              <span className="truncate">{candidate.name}</span>
            </WorkbenchAction>
          ))}
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-auto p-4">
        {file && (
          <div className="mb-3 flex items-center justify-between gap-3 font-mono text-micro text-muted-foreground">
            <span className="truncate text-foreground" title={file.relPath}>
              {file.relPath}
            </span>
            <span>Vega-Lite · MolPlot</span>
          </div>
        )}
        {loading && <p className="text-body text-muted-foreground">Loading plot…</p>}
        {error && (
          <EmptyState
            density="compact"
            icon={<BarChart3 className="size-5" />}
            title="Cannot render this observable"
            description={error}
          />
        )}
        {spec && (
          <div className="min-h-96 bg-surface/65 p-3">
            <MolplotRawChart
              spec={{ spec }}
              style={{ width: "100%", height: "var(--spacing-chart-lg)" }}
            />
          </div>
        )}
      </main>
    </div>
  );
};
