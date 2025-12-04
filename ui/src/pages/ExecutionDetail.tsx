import { useParams } from 'react-router-dom';
import { WorkflowEditor } from '@/components/workflow/WorkflowEditor';
import { type Node, type Edge } from '@xyflow/react';
import { Button } from '@/components/ui/button';
import { Play } from 'lucide-react';
import { executionApi } from '@/services/api';
import { useState } from 'react';

export const ExecutionDetail = () => {
  const { projectId, experimentId, runId } = useParams();
  const [isRunning, setIsRunning] = useState(false);

  // Mock data - in a real app, fetch based on ID
  const initialNodes: Node[] = [
    { id: '1', type: 'load-molecule', position: { x: 250, y: 50 }, data: { label: 'Load Aspirin' } },
    { id: '2', type: 'optimize-geometry', position: { x: 250, y: 200 }, data: { label: 'Optimize' } },
    { id: '3', type: 'calc-energy', position: { x: 250, y: 350 }, data: { label: 'Energy' } },
    { id: '4', type: 'save-results', position: { x: 250, y: 500 }, data: { label: 'Save' } },
  ];

  const initialEdges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e2-3', source: '2', target: '3' },
    { id: 'e3-4', source: '3', target: '4' },
  ];

  const executionStatus: Record<string, 'success' | 'failed' | 'pending'> = {
    'e1-2': 'success',
    'e2-3': runId === '2' ? 'failed' : 'success', // Fail for execution 2
    'e3-4': 'pending',
  };

  const handleRun = async () => {
    if (!projectId || !experimentId || !runId) return;

    try {
      setIsRunning(true);
      console.log("Starting execution...");
      
      await executionApi.start(projectId, experimentId, runId);
      
      console.log("Execution started");
    } catch (error) {
      console.error("Execution failed", error);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex items-center justify-between mb-4 px-4 pt-4">
        <h2 className="text-3xl font-bold tracking-tight">Execution #{runId}</h2>
        <Button onClick={handleRun} disabled={isRunning}>
          <Play className="mr-2 h-4 w-4" />
          {isRunning ? 'Running...' : 'Run Workflow'}
        </Button>
      </div>
      <div className="flex-1 border rounded-lg bg-muted/20 relative overflow-hidden m-4">
        <WorkflowEditor 
          readOnly={true} 
          initialNodes={initialNodes}
          initialEdges={initialEdges}
          executionStatus={executionStatus}
        />
      </div>
    </div>
  );
};
