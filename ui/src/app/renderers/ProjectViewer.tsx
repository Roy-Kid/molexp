import { Trash2, Folder, FlaskConical, Play, Archive,  Activity } from "lucide-react";
import { useState, useMemo, useEffect } from "react";
import { Button } from "@/components/ui/button";
import type { RendererProps, ApiAssetResponse } from "@/app/types";
import { CreateExperimentDialog } from "@/app/components/CreateExperimentDialog";
import { workspaceApi } from "@/app/state/api";
import { useUrlState } from "@/app/state/useUrlState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

const MetricStat = ({ label, value, icon: Icon, colorClass }: { label: string, value: number | string, icon: any, colorClass?: string }) => (
  <div className="flex flex-col gap-1">
     <div className="flex items-center gap-2 text-muted-foreground mb-1">
        <Icon className={`h-4 w-4 ${colorClass || "text-muted-foreground"}`} />
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
     </div>
     <span className="text-3xl font-light tracking-tight text-foreground">{value}</span>
  </div>
);

export const ProjectViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [projectAssets, setProjectAssets] = useState<ApiAssetResponse[]>([]);
  const { setSelection } = useUrlState();

  // Extract project name/id from selection
  const projectId = selection.objectId;
  const project = snapshot.projects.find((p) => p.id === projectId);
  
  // Fetch project assets
  useEffect(() => {
      if (projectId) {
          workspaceApi.getProjectAssets(projectId)
            .then(setProjectAssets)
            .catch(err => console.error("Failed to load project assets", err));
      }
  }, [projectId, onRefresh]);
  
  // Filter experiments and runs
  const projectExperiments = useMemo(() => 
    snapshot.experiments.filter(e => e.projectId === projectId),
  [snapshot.experiments, projectId]);

  const projectRuns = useMemo(() => 
      snapshot.runs.filter(r => r.projectId === projectId),
  [snapshot.runs, projectId]);

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete project "${projectId}"?`)) {
      return;
    }
    setIsDeleting(true);
    try {
      await workspaceApi.deleteProject(projectId);
      onRefresh(); 
    } catch (error) {
      console.error("Failed to delete project:", error);
      alert("Failed to delete project");
    } finally {
      setIsDeleting(false);
    }
  };

  const navigateToExperiment = (experimentId: string) => {
    setSelection({
        objectType: "experiment",
        objectId: experimentId,
    });
  };

  if (!project) return <div className="p-8 text-muted-foreground">Project not found.</div>;

  return (
    <div className="flex h-full flex-col bg-background">
        {/* Header Hero */}
        <div className="flex flex-col gap-6 px-8 py-8 border-b bg-background">
            <div className="flex items-start justify-between">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/10 rounded-lg">
                            <Folder className="h-6 w-6 text-blue-600" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">{project.name}</h1>
                        <Badge variant="outline" className="ml-2 font-mono text-xs text-muted-foreground">
                            {projectId.substring(0,8)}
                        </Badge>
                    </div>
                    <p className="text-muted-foreground max-w-2xl pl-[3.25rem]">
                        {project.summary || "No description provided."}
                    </p>
                </div>
                <div className="flex items-center gap-2">
                     <CreateExperimentDialog
                        projectId={projectId}
                        onExperimentCreated={onRefresh}
                     />
                     <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleDelete}
                        disabled={isDeleting}
                        className="text-muted-foreground hover:text-destructive transition-colors"
                        title="Delete Project"
                     >
                        <Trash2 className="h-5 w-5" />
                     </Button>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="flex items-center gap-12 pl-[3.25rem] py-2">
                <MetricStat 
                    label="Experiments" 
                    value={projectExperiments.length} 
                    icon={FlaskConical} 
                    colorClass="text-purple-500"
                />
                <div className="h-12 w-px bg-border/50" />
                <MetricStat 
                    label="Total Runs" 
                    value={projectRuns.length} 
                    icon={Activity} 
                    colorClass="text-emerald-500"
                />
                <div className="h-12 w-px bg-border/50" />
                <MetricStat 
                    label="Project Assets" 
                    value={projectAssets.length} 
                    icon={Archive} 
                    colorClass="text-amber-500"
                />
            </div>
        </div>

        {/* Content Tabs */}
        <div className="flex-1 overflow-hidden flex flex-col">
            <Tabs defaultValue="experiments" className="flex-1 flex flex-col">
                <div className="px-8 py-3 border-b bg-background">
                    <TabsList className="h-auto w-fit justify-start bg-muted/20 p-1 rounded-lg">
                        <TabsTrigger 
                            value="experiments" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Experiments
                        </TabsTrigger>
                        <TabsTrigger 
                            value="assets" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Assets
                        </TabsTrigger>
                        <TabsTrigger 
                            value="settings" 
                            className="px-4 py-2 rounded-md font-medium text-sm"
                        >
                            Settings
                        </TabsTrigger>
                    </TabsList>
                </div>

                <TabsContent value="experiments" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
                    <div className="flex-1 overflow-auto">
                         <table className="w-full text-sm text-left">
                            <thead className="bg-muted/20 text-muted-foreground font-medium sticky top-0 backdrop-blur-md">
                                <tr>
                                    <th className="py-3 px-6 w-auto">Experiment Name</th>
                                    <th className="py-3 px-6 w-[120px]">ID</th>
                                    <th className="py-3 px-6 w-[160px]">Status</th>
                                    <th className="py-3 px-6 w-[180px]">Updated</th>
                                    <th className="py-3 px-6 w-[60px] text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50">
                                {projectExperiments.length === 0 ? (
                                    <tr>
                                        <td colSpan={5} className="py-12 text-center text-muted-foreground">
                                            No experiments yet. Click "Create Experiment" to start.
                                        </td>
                                    </tr>
                                ) : (
                                    projectExperiments.map((exp) => (
                                        <tr 
                                            key={exp.id}
                                            className="group hover:bg-muted/40 transition-colors cursor-pointer"
                                            onClick={() => navigateToExperiment(exp.id)}
                                        >
                                            <td className="py-3 px-6">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-1.5 rounded-md bg-purple-500/10 text-purple-600 group-hover:bg-purple-500/20 transition-colors">
                                                        <FlaskConical className="h-4 w-4" />
                                                    </div>
                                                    <span className="font-medium text-foreground">{exp.name}</span>
                                                </div>
                                            </td>
                                            <td className="py-3 px-6 font-mono text-xs text-muted-foreground">{exp.id.substring(0,8)}</td>
                                            <td className="py-3 px-6">
                                                <Badge variant="secondary" className="font-normal">Active</Badge>
                                            </td>
                                            <td className="py-3 px-6 text-muted-foreground">
                                                {new Date(exp.updatedAt).toLocaleDateString()}
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

                <TabsContent value="assets" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-muted/20 text-muted-foreground font-medium sticky top-0 backdrop-blur-md">
                                <tr>
                                    <th className="py-3 px-6 w-auto">Asset Name</th>
                                    <th className="py-3 px-6 w-[150px]">Type</th>
                                    <th className="py-3 px-6 w-[120px]">Size</th>
                                    <th className="py-3 px-6 w-[180px]">Created</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50">
                                {projectAssets.length === 0 ? (
                                    <tr>
                                        <td colSpan={4} className="py-12 text-center text-muted-foreground">
                                            No assets found in this project.
                                        </td>
                                    </tr>
                                ) : (
                                    projectAssets.map((asset) => (
                                        <tr key={asset.id} className="hover:bg-muted/40 transition-colors">
                                            <td className="py-3 px-6 font-medium">
                                                <div className="flex items-center gap-3">
                                                     <div className="p-1.5 rounded-md bg-amber-500/10 text-amber-600">
                                                        <Archive className="h-4 w-4" />
                                                    </div>
                                                    {asset.assetId}
                                                </div>
                                            </td>
                                            <td className="py-3 px-6 text-muted-foreground">{asset.format}</td>
                                            <td className="py-3 px-6 font-mono text-xs">{asset.size} B</td>
                                            <td className="py-3 px-6 text-muted-foreground">
                                                {new Date(asset.created).toLocaleDateString()}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                         </table>
                    </div>
                </TabsContent>
                
                <TabsContent value="settings" className="flex-1 p-6">
                    <div className="max-w-xl space-y-4">
                        <div className="p-4 rounded-lg border bg-card text-card-foreground shadow-sm">
                            <h3 className="text-lg font-medium mb-2">Project Settings</h3>
                            <p className="text-sm text-muted-foreground mb-4">
                                Manage project-level configurations and danger zones.
                            </p>
                            <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete Project
                            </Button>
                        </div>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    </div>
  );
};
