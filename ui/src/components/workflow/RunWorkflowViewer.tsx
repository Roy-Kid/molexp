import React, { useEffect, useState } from 'react';
import { WorkflowEditor } from './WorkflowEditor';
import { API_ENDPOINTS } from '@/config/api';
import type { TaskGraphJson } from '@/types/task_graph_ir';
import { Loader } from 'lucide-react';

interface RunWorkflowViewerProps {
  projectId: string;
  experimentId: string;
  runId: string;
}

export const RunWorkflowViewer: React.FC<RunWorkflowViewerProps> = ({ projectId, experimentId, runId }) => {
  const [graphData, setGraphData] = useState<TaskGraphJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRunData = async () => {
      try {
        setLoading(true);
        const response = await fetch(API_ENDPOINTS.runs.get(projectId, experimentId, runId));
        if (!response.ok) throw new Error('Failed to fetch run data');
        
        const runData = await response.json();
        
        // Check both locations for robustness
        const serializedGraph = runData.workflow?.serializedGraph || runData.workflowSnapshot?.serializedGraph;
        
        if (serializedGraph) {
          // Parse JSON IR string if it's a string, or use directly if object
          const data: TaskGraphJson = typeof serializedGraph === 'string' 
            ? JSON.parse(serializedGraph)
            : serializedGraph;
            
          setGraphData(data);
        } else {
          // Fallback if no graph data
          setGraphData({
            name: 'Unknown Workflow',
            nodes: [
              { id: '1', type: 'process', label: runData.workflow?.file || 'Unknown Workflow', params: {} }
            ],
            edges: []
          });
        }
      } catch (err) {
        console.error(err);
        setError('Failed to load workflow visualization');
      } finally {
        setLoading(false);
      }
    };

    fetchRunData();
  }, [projectId, experimentId, runId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-muted/10">
        <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-500">
        {error}
      </div>
    );
  }

  return (
    <div className="h-full border rounded-lg overflow-hidden bg-background">
      {graphData && (
        <WorkflowEditor 
          readOnly={true}
          initialGraph={graphData}
        />
      )}
    </div>
  );
};
