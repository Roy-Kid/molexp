import { FileText, CheckCircle2, XCircle, AlertCircle, Clock, Activity, Box, Terminal } from "lucide-react";
import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import type { RendererProps, SemanticStatus } from "@/app/types";
import { buildMetadataFields } from "@/app/renderers/metadata";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const MetricStat = ({ label, value, icon: Icon, colorClass }: { label: string, value: number | string, icon: any, colorClass?: string }) => (
  <div className="flex flex-col gap-1">
     <div className="flex items-center gap-2 text-muted-foreground mb-1">
        <Icon className={`h-4 w-4 ${colorClass || "text-muted-foreground"}`} />
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
     </div>
     <span className="text-3xl font-light tracking-tight text-foreground">{value}</span>
  </div>
);

const StatusBadge = ({ status }: { status: SemanticStatus }) => {
  if (status === "succeeded") {
    return (
      <Badge variant="default" className="bg-green-600 hover:bg-green-700 gap-1.5 pl-1.5 pr-2.5 py-1">
        <CheckCircle2 className="h-4 w-4" /> Succeeded
      </Badge>
    );
  }
  if (status === "failed") {
    return (
      <Badge variant="destructive" className="gap-1.5 pl-1.5 pr-2.5 py-1">
        <XCircle className="h-4 w-4" /> Failed
      </Badge>
    );
  }
  if (status === "running") {
    return (
      <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200 gap-1.5 pl-1.5 pr-2.5 py-1 animate-pulse">
        <Activity className="h-4 w-4" /> Running
      </Badge>
    );
  }
  if (status === "cancelled") {
    return (
      <Badge variant="secondary" className="gap-1.5 pl-1.5 pr-2.5 py-1">
        <AlertCircle className="h-4 w-4" /> Cancelled
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1.5 pl-1.5 pr-2.5 py-1">
      <Clock className="h-4 w-4" /> {status}
    </Badge>
  );
};

export const RunViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);
  
  const run = useMemo(() => {
     return snapshot.runs.find(r => r.id === selection.objectId);
  }, [snapshot.runs, selection.objectId]);

  if (!run) {
    return <div className="p-8 text-muted-foreground">Run not found.</div>;
  }

  // Filter interesting fields to show in the table
  const displayFields = fields.filter(f => 
    !["Run", "Status", "Summary", "Project", "Experiment"].includes(f.label)
  );

  return (
    <div className="flex h-full flex-col bg-background">
        {/* Header Hero */}
        <div className="flex flex-col gap-6 px-8 py-8 border-b bg-background">
            <div className="flex items-start justify-between">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/10 rounded-lg">
                            <FileText className="h-6 w-6 text-blue-600" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">{run.name}</h1>
                        <Badge variant="outline" className="font-mono text-xs text-muted-foreground ml-2">
                            {run.id}
                        </Badge>
                    </div>
                     <div className="flex items-center gap-2 pl-[3.25rem]">
                         <StatusBadge status={run.status} />
                     </div>
                </div>
                {/* Placeholder for Actions */}
            </div>

            {/* Metrics Row */}
            <div className="flex items-center gap-12 pl-[3.25rem] py-2">
                <MetricStat 
                    label="Updated" 
                    value={new Date(run.updatedAt).toLocaleTimeString()} 
                    icon={Clock} 
                />
                 {/* Placeholders for future metrics */}
            </div>
        </div>

        {/* Content Tabs */}
        <div className="flex-1 overflow-hidden flex flex-col">
            <Tabs defaultValue="overview" className="flex-1 flex flex-col">
                 <div className="px-8 py-3 border-b bg-background">
                    <TabsList className="h-auto w-fit justify-start bg-muted/20 p-1 rounded-lg">
                        <TabsTrigger 
                            value="overview" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Overview
                        </TabsTrigger>
                        <TabsTrigger 
                            value="logs" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Logs
                        </TabsTrigger>
                         <TabsTrigger 
                            value="artifacts" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Artifacts
                        </TabsTrigger>
                    </TabsList>
                </div>

                <TabsContent value="overview" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
                    <div className="flex-1 overflow-auto">
                         <table className="w-full text-sm text-left">
                            <tbody className="divide-y divide-border/50">
                                {displayFields.length === 0 ? (
                                    <tr>
                                        <td colSpan={2} className="py-12 text-center text-muted-foreground">
                                            No additional details.
                                        </td>
                                    </tr>
                                ) : (
                                    displayFields.map((field, i) => (
                                        <tr key={i} className="hover:bg-muted/30 transition-colors">
                                            <td className="py-3 px-6 font-medium text-muted-foreground w-[200px]">
                                                {field.label}
                                            </td>
                                            <td className="py-3 px-6 font-mono text-xs">
                                                {field.value}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                        
                        <div className="p-6">
                            <h3 className="text-sm font-medium text-muted-foreground mb-3 uppercase tracking-wider">Summary</h3>
                             <div className="p-4 rounded-lg bg-muted/30 border text-sm">
                                {run.summary || "No summary provided."}
                            </div>
                        </div>
                    </div>
                </TabsContent>

                <TabsContent value="logs" className="flex-1 p-0 flex flex-col bg-slate-950 text-slate-50">
                     <div className="flex items-center px-4 py-2 border-b border-slate-800 bg-slate-900 text-xs font-mono text-slate-400">
                        <Terminal className="h-3 w-3 mr-2" />
                        stdout/stderr
                     </div>
                     <div className="flex-1 p-4 font-mono text-xs overflow-auto">
                        <div className="opacity-50 italic">Log streaming not implemented yet.</div>
                     </div>
                </TabsContent>

                 <TabsContent value="artifacts" className="flex-1 p-0 flex flex-col">
                     <div className="flex-1 p-8 flex flex-col items-center justify-center text-muted-foreground">
                        <Box className="h-12 w-12 mb-4 opacity-20" />
                        <div>No artifacts generated.</div>
                     </div>
                </TabsContent>
            </Tabs>
        </div>
    </div>
  );
};
