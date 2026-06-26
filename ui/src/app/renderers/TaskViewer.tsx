import { ArrowDown, ArrowRight, ArrowUp, FileText } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, RendererProps, TaskSelection } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

/**
 * TaskViewer — the right-inspector panel shown when a workflow-graph node is
 * clicked. It renders *in place* over the current run page (the graph stays in
 * the center); it is never a standalone navigable page. Shows the task's
 * identity, its place in the DAG (upstream → this → downstream, each clickable
 * to re-pin the inspector), and the run assets it produced.
 */
const taskProducerId = (asset: ApiAssetResponse): string | undefined =>
  (asset.producer as Record<string, unknown> | null | undefined)?.task_id as string | undefined;

const formatConfigValue = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const Section = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): JSX.Element => (
  <div className="px-3 py-2.5">
    <h3 className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
      {title}
    </h3>
    {children}
  </div>
);

export const TaskViewer = ({ selection, snapshot }: RendererProps): JSX.Element | null => {
  const { setSelection } = useNavigationState(snapshot);
  const { inspectTask, clearInspectedTask } = useInspectedTask();
  const [assets, setAssets] = useState<ApiAssetResponse[]>([]);

  const task = selection.objectType === "task" ? (selection as TaskSelection) : null;
  const runId = task?.runId ?? "";
  const taskId = task?.taskId ?? "";

  useEffect(() => {
    let cancelled = false;
    if (!runId) {
      setAssets([]);
      return;
    }
    workspaceApi
      .getRunAssets(runId)
      .then((items) => {
        if (!cancelled) setAssets(items);
      })
      .catch(() => {
        if (!cancelled) setAssets([]);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const run = useMemo(() => snapshot.runs.find((r) => r.id === runId), [snapshot.runs, runId]);
  const workflow = useMemo(() => {
    // From a run page, scope to the run's experiment; in a compiled preview
    // (no run) resolve the owning workflow by which graph contains the node.
    if (run) return snapshot.workflows.find((w) => w.experimentId === run.experimentId) ?? null;
    return (
      snapshot.workflows.find((w) => w.graph?.task_configs.some((n) => n.id === taskId)) ?? null
    );
  }, [snapshot.workflows, run, taskId]);
  const node = workflow?.graph?.task_configs.find((n) => n.id === taskId) ?? null;
  const links = workflow?.graph?.links ?? [];
  const upstream = links.filter((e) => e.to === taskId).map((e) => e.from);
  const downstream = links.filter((e) => e.from === taskId).map((e) => e.to);
  const products = useMemo(
    () => assets.filter((a) => taskProducerId(a) === taskId),
    [assets, taskId],
  );

  if (!task) return null;

  const TaskChips = ({ ids }: { ids: string[] }): JSX.Element =>
    ids.length === 0 ? (
      <span className="text-xs text-muted-foreground">none</span>
    ) : (
      <div className="flex flex-wrap gap-1.5">
        {ids.map((id) => (
          <Button
            key={id}
            variant="outline"
            size="sm"
            className="h-7 px-2 font-mono text-xs"
            onClick={() => inspectTask(id, runId)}
          >
            {id}
          </Button>
        ))}
      </div>
    );

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between gap-2 border-b border-border/70 bg-muted/20 px-3 py-1.5">
        <h2 className="truncate text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Task
        </h2>
        {node?.type && (
          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] uppercase tracking-wide">
            {node.type}
          </Badge>
        )}
      </div>

      <div className="flex-1 divide-y divide-border/50 overflow-auto">
        <Section title="Identity">
          <p className="truncate font-mono text-sm font-semibold text-foreground">{taskId}</p>
          <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
            [{node?.label ?? node?.type ?? "—"}]
          </p>
          {run && (
            <Button
              variant="link"
              size="sm"
              className="mt-1 h-auto p-0 text-xs"
              onClick={clearInspectedTask}
            >
              ← {run.name ?? run.id}
            </Button>
          )}
        </Section>

        {node?.source ? (
          <Section title="Source">
            <pre className="max-h-80 overflow-auto rounded-md border border-border/60 bg-muted/30 p-2.5 font-mono text-[11px] leading-relaxed text-foreground">
              <code>{node.source}</code>
            </pre>
          </Section>
        ) : (
          <Section title="Source">
            <p className="text-xs italic text-muted-foreground">
              No source captured for this node.
            </p>
          </Section>
        )}

        {node?.config && Object.keys(node.config).length > 0 && (
          <Section title="Inputs">
            <dl className="space-y-1">
              {Object.entries(node.config).map(([key, value]) => (
                <div key={key} className="flex gap-2 text-xs">
                  <dt className="flex-none font-medium text-muted-foreground">{key}</dt>
                  <dd className="min-w-0 flex-1 break-all text-right font-mono text-foreground">
                    {formatConfigValue(value)}
                  </dd>
                </div>
              ))}
            </dl>
          </Section>
        )}

        <Section title="Upstream">
          <div className="flex items-center gap-1.5">
            <ArrowUp className="h-3 w-3 flex-none text-muted-foreground" />
            <TaskChips ids={upstream} />
          </div>
        </Section>

        <Section title="Downstream">
          <div className="flex items-center gap-1.5">
            <ArrowDown className="h-3 w-3 flex-none text-muted-foreground" />
            <TaskChips ids={downstream} />
          </div>
        </Section>

        <Section title={`Products (${products.length})`}>
          {products.length === 0 ? (
            <p className="text-xs italic text-muted-foreground">No assets published.</p>
          ) : (
            <div className="space-y-1.5">
              {products.map((asset) => (
                <button
                  key={asset.id}
                  type="button"
                  className="flex w-full items-start gap-2 rounded-md border border-border/70 bg-muted/20 p-2 text-left transition-colors hover:border-border hover:bg-muted/40"
                  onClick={() => setSelection({ objectType: "asset", objectId: asset.id })}
                >
                  <FileText className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium text-foreground">{asset.name}</div>
                    <div className="truncate font-mono text-[10px] text-muted-foreground">
                      {asset.path}
                    </div>
                  </div>
                  <ArrowRight className="mt-0.5 h-3.5 w-3.5 flex-none text-muted-foreground" />
                </button>
              ))}
            </div>
          )}
        </Section>
      </div>
    </div>
  );
};
