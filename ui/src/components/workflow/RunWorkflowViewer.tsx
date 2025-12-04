import React, { useEffect, useState, useCallback } from 'react';
import { WorkflowEditor } from './WorkflowEditor';
import { NodeDetailPanel } from './NodeDetailPanel';
import { API_ENDPOINTS } from '@/config/api';
import type { TaskGraphJson } from '@/types/task_graph_ir';
import { Loader } from 'lucide-react';
import type { Node } from '@xyflow/react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';

interface RunWorkflowViewerProps {
  projectId: string;
  experimentId: string;
  runId: string;
}

export const RunWorkflowViewer: React.FC<RunWorkflowViewerProps> = ({ projectId, experimentId, runId }) => {
  const [graphData, setGraphData] = useState<TaskGraphJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [executionStatus, setExecutionStatus] = useState<Record<string, 'success' | 'failed' | 'pending' | 'running'>>({});
  const [logs, setLogs] = useState<Record<string, string[]>>({});

  const fetchRunData = useCallback(async (isPolling = false) => {
    try {
      if (!isPolling) setLoading(true);
      const response = await fetch(API_ENDPOINTS.runs.get(projectId, experimentId, runId));
      if (!response.ok) throw new Error('Failed to fetch run data');
      
      const runData = await response.json();
      
      // Update graph data only if not already set (to avoid re-layouting)
      if (!graphData) {
        const serializedGraph = runData.workflow?.serializedGraph || runData.workflowSnapshot?.serializedGraph;
        
        if (serializedGraph) {
          const data: TaskGraphJson = typeof serializedGraph === 'string' 
            ? JSON.parse(serializedGraph)
            : serializedGraph;
          setGraphData(data);
        } else {
          setGraphData({
            name: 'Unknown Workflow',
            nodes: [
              { id: '1', type: 'process', label: runData.workflow?.file || 'Unknown Workflow', params: {} }
            ],
            edges: []
          });
        }
      }

      // Update execution status
      // Assuming runData has a status field or a tasks map
      // For now, let's mock some status updates based on the run status if detailed task status isn't available
      // In a real implementation, we'd expect runData.task_statuses = { "node_id": "success", ... }
      
      if (runData.task_statuses) {
        setExecutionStatus(runData.task_statuses);
      } else {
        // Fallback/Mock for demonstration if backend doesn't send node-level status yet
        // This logic mimics a "completed" run where everything is green, or a "running" run
        const statusMap: Record<string, any> = {};
        if (graphData) {
            graphData.nodes.forEach(node => {
                statusMap[node.id] = runData.status === 'completed' ? 'success' : 
                                     runData.status === 'failed' ? 'failed' : 'pending';
            });
        }
        setExecutionStatus(statusMap);
      }
      
      // Update logs if available
      if (runData.logs) {
          setLogs(runData.logs);
      }

    } catch (err) {
      console.error(err);
      if (!isPolling) setError('Failed to load workflow visualization');
    } finally {
      if (!isPolling) setLoading(false);
    }
  }, [projectId, experimentId, runId, graphData]);

  // Initial fetch
  useEffect(() => {
    fetchRunData();
  }, [fetchRunData]);

  // Polling
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchRunData(true);
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(intervalId);
  }, [fetchRunData]);

  if (loading && !graphData) {
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
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel defaultSize={selectedNode ? 70 : 100} minSize={30}>
            {graphData && (
                <WorkflowEditor 
                readOnly={true}
                initialGraph={graphData}
                executionStatus={executionStatus}
                onNodeSelect={setSelectedNode}
                />
            )}
        </ResizablePanel>
        
        {selectedNode && (
            <>
                <ResizableHandle />
                <ResizablePanel defaultSize={30} minSize={20} maxSize={50}>
                    <NodeDetailPanel 
                        node={selectedNode} 
                        onClose={() => setSelectedNode(null)}
                        executionStatus={executionStatus[selectedNode.id]}
                        logs={logs[selectedNode.id]}
                    />
                </ResizablePanel>
            </>
        )}
      </ResizablePanelGroup>
    </div>
  );
};
