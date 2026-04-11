import { Clock, FlaskConical, Play, Trash2, Activity, CheckCircle2, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { RendererProps, SemanticStatus } from "@/app/types";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import { workspaceApi } from "@/app/state/api";
import { useUrlState } from "@/app/state/useUrlState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SnapshotDiffPanel } from "@/app/renderers/SnapshotViewer";

const MetricStat = ({ label, value, icon: Icon, colorClass }: { label: string, value: number | string, icon: any, colorClass?: string }) => (
  <div className="flex flex-col gap-1">
     <div className="flex items-center gap-2 text-muted-foreground mb-1">
        <Icon className={`h-4 w-4 ${colorClass || "text-muted-foreground"}`} />
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
     </div>
     <span className="text-3xl font-light tracking-tight text-foreground">{value}</span>
  </div>
);

const StatusCell = ({ status }: { status: SemanticStatus }) => {
    const styles = {
        succeeded: "bg-green-500/10 text-green-700 hover:bg-green-500/20",
        failed: "bg-red-500/10 text-red-700 hover:bg-red-500/20",
        running: "bg-blue-500/10 text-blue-700 hover:bg-blue-500/20 animate-pulse",
        pending: "bg-muted text-muted-foreground",
        cancelled: "bg-muted text-muted-foreground strike-through",
        skipped: "bg-amber-500/10 text-amber-700",
        active: "bg-muted",
        archived: "bg-muted",
        draft: "bg-amber-500/10 text-amber-700" 
    };
    return (
        <Badge variant="secondary" className={`font-normal ${styles[status] || styles.pending}`}>
            {status}
        </Badge>
    );
};

export const ExperimentViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const { setSelection } = useUrlState();

  // Find the experiment in snapshot
  const experimentId = selection.objectId;
  const experiment = snapshot.experiments.find((e) => e.id === experimentId);
  const projectId = experiment?.projectId || "";
  
  // Filter runs for this experiment
  const runs = useMemo(() => 
    snapshot.runs.filter(r => r.experimentId === experimentId), 
  [snapshot.runs, experimentId]);

  const stats = useMemo(() => {
      return {
          total: runs.length,
          succeeded: runs.filter(r => r.status === "succeeded").length,
          failed: runs.filter(r => r.status === "failed").length,
          running: runs.filter(r => r.status === "running").length,
      };
  }, [runs]);

  const handleDelete = async () => {
    if (!projectId) return;
    if (!confirm(`Are you sure you want to delete experiment "${experimentId}"?`)) {
      return;
    }
    setIsDeleting(true);
    try {
      await workspaceApi.deleteExperiment(projectId, experimentId);
      onRefresh(); 
    } catch (error) {
      console.error("Failed to delete experiment:", error);
      alert("Failed to delete experiment");
    } finally {
      setIsDeleting(false);
    }
  };

  const navigateToRun = (runId: string) => {
    setSelection({
        objectType: "run",
        objectId: runId,
    });
  };

  if (!experiment || !projectId) {
    return <div className="p-8 text-muted-foreground">Experiment not found.</div>;
  }

  return (
    <div className="flex h-full flex-col bg-background">
        {/* Header Hero */}
        <div className="flex flex-col gap-6 px-8 py-8 border-b bg-background">
            <div className="flex items-start justify-between">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-purple-500/10 rounded-lg">
                            <FlaskConical className="h-6 w-6 text-purple-600" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">{experiment.name}</h1>
                    </div>
                    <div className="flex items-center gap-2 pl-[3.25rem]">
                        <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                            {experimentId.substring(0,8)}
                        </Badge>
                        <span className="text-muted-foreground">•</span>
                        <button 
                             className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-foreground hover:bg-muted/80 transition-colors cursor-pointer"
                             onClick={() => {
                                 // Assuming the workflow ID matches the name or we can derive it.
                                 // For now using the filename as the ID or finding it from snapshot if possible.
                                 // In the mock data, workflow ID usually matches or is referenced.
                                 // But here we only have the filename string.
                                 // Let's try to find a workflow with this name/file in snapshot 
                                 // or just navigate to a workflow object with this ID.
                                 const workflow = snapshot.workflows.find(w => w.name === experiment.workflowFile || w.id === experiment.workflowFile);
                                 if (workflow) {
                                     setSelection({
                                         objectType: "workflow",
                                         objectId: workflow.id,
                                         workflowId: workflow.id
                                     });
                                 } else {
                                     alert(`Workflow "${experiment.workflowFile}" not found in workspace.`);
                                 }
                             }}
                        >
                            {experiment.workflowFile || "workflow.json"}
                        </button>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                   <CreateRunDialog
                        projectId={projectId}
                        experimentId={experimentId}
                        workflowFile={experiment?.workflowFile || ""} 
                        onRunCreated={onRefresh}
                      />
                     <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleDelete}
                        disabled={isDeleting}
                        className="text-muted-foreground hover:text-destructive transition-colors"
                        title="Delete Experiment"
                     >
                        <Trash2 className="h-5 w-5" />
                     </Button>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="flex items-center gap-10 pl-[3.25rem] py-2">
                <MetricStat 
                    label="Runs" 
                    value={stats.total} 
                    icon={Activity} 
                    colorClass="text-foreground"
                />
                <div className="h-10 w-px bg-border/50" />
                <MetricStat 
                    label="Succeeded" 
                    value={stats.succeeded} 
                    icon={CheckCircle2} 
                    colorClass="text-green-500"
                />
                <div className="h-10 w-px bg-border/50" />
                <MetricStat 
                    label="Failed" 
                    value={stats.failed} 
                    icon={XCircle} 
                    colorClass="text-red-500"
                />
                 <div className="h-10 w-px bg-border/50" />
                <MetricStat 
                    label="Running" 
                    value={stats.running} 
                    icon={Clock} 
                    colorClass="text-blue-500"
                />
            </div>
        </div>

        {/* Content Tabs */}
        <div className="flex-1 overflow-hidden flex flex-col">
            <Tabs defaultValue="runs" className="flex-1 flex flex-col">
                 <div className="px-8 py-3 border-b bg-background">
                    <TabsList className="h-auto w-fit justify-start bg-muted/20 p-1 rounded-lg">
                        <TabsTrigger 
                            value="runs" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Runs
                        </TabsTrigger>
                        <TabsTrigger
                            value="details"
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Details
                        </TabsTrigger>
                        <TabsTrigger
                            value="diff"
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Diff
                        </TabsTrigger>
                    </TabsList>
                </div>

                <TabsContent value="runs" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
                     <div className="flex-1 overflow-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-muted/20 text-muted-foreground font-medium sticky top-0 backdrop-blur-md">
                                <tr>
                                    <th className="py-3 px-6 w-[120px]">Run ID</th>
                                    <th className="py-3 px-6 w-[140px]">Status</th>
                                    <th className="py-3 px-6 w-auto">Summary</th>
                                    <th className="py-3 px-6 w-[180px]">Updated</th>
                                    <th className="py-3 px-6 w-[60px] text-right">Act</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50">
                                {runs.length === 0 ? (
                                    <tr>
                                        <td colSpan={5} className="py-12 text-center text-muted-foreground">
                                            No runs yet. Click "Start Run" to begin.
                                        </td>
                                    </tr>
                                ) : (
                                    runs.map((run) => (
                                        <tr 
                                            key={run.id}
                                            className="group hover:bg-muted/40 transition-colors cursor-pointer"
                                            onClick={() => navigateToRun(run.id)}
                                        >
                                            <td className="py-3 px-6 font-mono text-xs text-muted-foreground">
                                                {run.id.substring(0, 8)}
                                            </td>
                                            <td className="py-3 px-6">
                                                <StatusCell status={run.status} />
                                            </td>
                                            <td className="py-3 px-6 text-muted-foreground truncate max-w-[200px]" title={run.summary}>
                                                {run.summary || "-"}
                                            </td>
                                            <td className="py-3 px-6 text-muted-foreground text-xs">
                                                {new Date(run.updatedAt).toLocaleString()}
                                            </td>
                                            <td className="py-3 px-6 text-right">
                                                 <Button size="icon" variant="ghost" className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <Play className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                                                </Button>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </TabsContent>

                <TabsContent value="details" className="flex-1 p-6 overflow-auto">
                    <pre className="text-xs bg-muted/50 p-4 rounded-md overflow-auto font-mono">
                        {JSON.stringify(experiment, null, 2)}
                    </pre>
                </TabsContent>

                <TabsContent value="diff" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
                    <SnapshotDiffPanel experimentRunIds={runs.map(r => r.id)} />
                </TabsContent>
            </Tabs>
        </div>
    </div>
  );
};
