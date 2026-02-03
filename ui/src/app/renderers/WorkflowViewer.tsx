import { Badge } from "@/components/ui/badge";
import { Workflow } from "lucide-react";
import type { RendererProps } from "@/app/types";

export const WorkflowViewer = ({
  selection,
  snapshot,
}: RendererProps): JSX.Element => {
  const workflowId = selection.objectId;
  const workflow = snapshot.workflows.find((w) => w.id === workflowId);

  // In a real implementation we would fetch the content. 
  // For now, let's assume we can get it or show a placeholder/mock.
  // Since the snapshot only has metadata, we might need to fetch the file content via API.
  // However, the current 'workspace-file' selection uses a different pattern.
  // We can try to assume the workflow ID is the file path or similar, OR just show the metadata.
  
  // NOTE: The previous design discussion assumed we could just show the content.
  // If we don't have the content in snapshot, we should use a "loading" state or fetch it.
  // Given constraints, I will use a placeholder or check if I can fetch it.
  // `workspaceApi.readFile` exists in `api.ts` (deduced from App.tsx handlers). 
  // But inside a renderer, we might want to just show what we have.
  
  if (!workflow) {
    return <div className="p-8 text-muted-foreground">Workflow not found.</div>;
  }

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex flex-col gap-6 px-8 py-8 border-b bg-background">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Workflow className="h-6 w-6 text-blue-600" />
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-foreground">{workflow.name}</h1>
            </div>
            <div className="flex items-center gap-2 pl-[3.25rem]">
              <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                {workflowId}
              </Badge>
            </div>
          </div>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden p-4">
        <p className="text-sm text-muted-foreground mb-4">
           Workflow previews are read-only.
        </p>
        <div className="h-full border rounded-md overflow-hidden">
             {/* 
                Ideally we would load the file content here. 
                Since we are in the UI layer and might not have async data fetching ready for this component
                without more boilerplate, I'll use a text placeholder or simple valid YAML structure.
             */}
             <div className="p-4 font-mono text-sm">
                # Workflow: {workflow.name}<br/>
                # ID: {workflow.id}<br/>
                <br/>
                steps:<br/>
                &nbsp;&nbsp;- name: example_step<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;image: python:3.9<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;command: echo "Hello World"<br/>
             </div>
        </div>
      </div>
    </div>
  );
};
