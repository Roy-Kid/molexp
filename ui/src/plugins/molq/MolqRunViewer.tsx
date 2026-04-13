import {
  Activity,
  Boxes,
  Clock3,
  Cpu,
  ExternalLink,
  ServerCog,
  TerminalSquare,
} from "lucide-react";
import { useMemo } from "react";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { RunViewer } from "@/app/renderers/RunViewer";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const getExecutorEntry = (executorInfo: Record<string, string>, ...keys: string[]): string | null => {
  for (const key of keys) {
    const value = executorInfo[key];
    if (value) {
      return value;
    }
  }
  return null;
};

export const MolqRunViewer = (props: RendererProps): JSX.Element => {
  const run = useMemo(() => {
    return props.snapshot.runs.find((item) => item.id === props.selection.objectId) ?? null;
  }, [props.selection.objectId, props.snapshot.runs]);

  if (!run) {
    return <div className="p-8 text-muted-foreground">Run not found.</div>;
  }

  if (run.executorInfo.backend !== "molq") {
    return <RunViewer {...props} />;
  }

  const scheduler = getExecutorEntry(run.executorInfo, "scheduler") ?? "unknown";
  const cluster = getExecutorEntry(run.executorInfo, "cluster_name", "cluster") ?? "default";
  const jobId = getExecutorEntry(run.executorInfo, "job_id", "molq_job_id") ?? "pending";
  const schedulerJobId =
    getExecutorEntry(run.executorInfo, "scheduler_job_id", "slurm_job_id") ?? "not assigned";

  const details = Object.entries(run.executorInfo);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="border-b border-border/60 bg-slate-950 px-8 py-8 text-slate-50">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="rounded-xl border border-cyan-400/30 bg-cyan-500/10 p-3">
                <ServerCog className="h-6 w-6 text-cyan-300" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-3xl font-semibold tracking-tight">{run.name}</h1>
                  <Badge className="bg-cyan-400/15 text-cyan-100 hover:bg-cyan-400/20">molq</Badge>
                </div>
                <p className="mt-1 text-sm text-slate-300">
                  Scheduler-backed execution routed through molq.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-400">
              <span>{scheduler}</span>
              <span className="text-slate-600">/</span>
              <span>{cluster}</span>
              <span className="text-slate-600">/</span>
              <span>{run.status}</span>
            </div>
          </div>

          <div className="grid min-w-[320px] grid-cols-2 gap-3">
            <Card className="border-cyan-500/20 bg-slate-900/80 text-slate-50">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm font-medium text-slate-300">
                  <Cpu className="h-4 w-4 text-cyan-300" />
                  Scheduler
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{scheduler}</CardContent>
            </Card>
            <Card className="border-cyan-500/20 bg-slate-900/80 text-slate-50">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm font-medium text-slate-300">
                  <Boxes className="h-4 w-4 text-cyan-300" />
                  Cluster
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{cluster}</CardContent>
            </Card>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="overview" className="flex h-full flex-col">
          <div className="border-b border-border/60 px-8 py-3">
            <TabsList className="h-auto w-fit rounded-lg bg-muted/30 p-1">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="scheduler">Scheduler</TabsTrigger>
              <TabsTrigger value="snapshot">Snapshot</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="m-0 flex-1 overflow-auto p-8">
            <div className="grid gap-4 lg:grid-cols-3">
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <TerminalSquare className="h-4 w-4 text-cyan-600" />
                    Molq Job
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <p className="font-mono text-sm">{jobId}</p>
                  <p className="text-xs text-muted-foreground">Logical molq job identifier.</p>
                </CardContent>
              </Card>
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <ExternalLink className="h-4 w-4 text-cyan-600" />
                    Scheduler Job
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <p className="font-mono text-sm">{schedulerJobId}</p>
                  <p className="text-xs text-muted-foreground">
                    Native scheduler identifier exposed by the backend.
                  </p>
                </CardContent>
              </Card>
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Clock3 className="h-4 w-4 text-cyan-600" />
                    Last Update
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <p className="text-sm font-medium">
                    {new Date(run.updatedAt).toLocaleString()}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Latest timestamp available in the workspace snapshot.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="mt-6 grid gap-4 lg:grid-cols-[1.5fr_1fr]">
              <Card className="border-border/60">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Activity className="h-4 w-4 text-cyan-600" />
                    Dispatch Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-sm">
                  <p className="text-muted-foreground">
                    {run.summary || "This run is managed by the molq execution backend."}
                  </p>
                  <div className="rounded-lg border border-dashed border-border/60 bg-muted/20 p-4">
                    <div className="grid gap-2 sm:grid-cols-2">
                      <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          Backend
                        </p>
                        <p className="font-medium">{run.executorInfo.backend}</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          Status
                        </p>
                        <p className="font-medium">{run.status}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-border/60">
                <CardHeader>
                  <CardTitle className="text-base">Normalized Executor Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {details.map(([key, value]) => (
                    <div key={key} className="space-y-1">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        {key}
                      </p>
                      <p className="font-mono text-xs text-foreground">{value}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="scheduler" className="m-0 flex-1 overflow-auto p-8">
            <Card className="border-border/60">
              <CardHeader>
                <CardTitle>Scheduler Metadata</CardTitle>
              </CardHeader>
              <CardContent className="overflow-auto">
                <table className="w-full text-left text-sm">
                  <tbody className="divide-y divide-border/50">
                    {details.map(([key, value]) => (
                      <tr key={key}>
                        <td className="w-[220px] py-3 font-medium text-muted-foreground">{key}</td>
                        <td className="py-3 font-mono text-xs">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="snapshot" className="m-0 flex-1 overflow-hidden">
            <RunSnapshotPanel runId={run.id} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
