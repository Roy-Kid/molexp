/**
 * Workflow Preview Component
 * 
 * Renders workflow (.flow) files using the visual workflow editor in read-only mode.
 * Parses the JSON content and displays the workflow graph.
 */

import React, { useState, useEffect } from 'react';
import { Loader, FileJson, AlertCircle } from 'lucide-react';
import { WorkflowEditor } from '@/components/workflow/WorkflowEditor';
import type { FilePreviewContentProps } from '@/lib/file-preview-plugins';
import type { TaskGraphJson } from '@/types/task_graph_ir';

/**
 * WorkflowPreview renders workflow IR files as an interactive graph.
 * Uses the WorkflowEditor component in read-only mode.
 */
export const WorkflowPreview: React.FC<FilePreviewContentProps> = ({
  content,
  name,
}) => {
  const [graph, setGraph] = useState<TaskGraphJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      const parsed = JSON.parse(content);
      setGraph(parsed);
      setError(null);
    } catch (e) {
      setError('Invalid workflow JSON format');
      setGraph(null);
    }
  }, [content]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <AlertCircle className="h-12 w-12 text-destructive mb-4" />
        <p className="font-medium text-destructive">Failed to parse workflow</p>
        <p className="text-sm text-muted-foreground mt-2">{error}</p>
        <div className="mt-6 p-4 bg-muted rounded-lg max-w-md overflow-auto">
          <pre className="text-xs text-left whitespace-pre-wrap">{content.slice(0, 500)}...</pre>
        </div>
      </div>
    );
  }

  if (!graph) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="h-full w-full flex flex-col">
      {/* Header */}
      <div className="border-b px-4 py-3 flex items-center gap-3 bg-background/95">
        <FileJson className="h-5 w-5 text-primary" />
        <div>
          <h3 className="font-medium">{name}</h3>
          <p className="text-xs text-muted-foreground">
            Workflow Preview • {graph.nodes?.length || 0} nodes
          </p>
        </div>
      </div>
      
      {/* Workflow Editor in read-only mode */}
      <div className="flex-1 overflow-hidden">
        <WorkflowEditor
          initialGraph={graph}
          readOnly={true}
        />
      </div>
    </div>
  );
};

export default WorkflowPreview;
