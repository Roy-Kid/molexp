import React, { useEffect, useState } from 'react';
import { Loader, FileJson } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { API_ENDPOINTS } from '@/config/api';
import { WorkflowEditor } from '@/components/workflow/WorkflowEditor';
import type { TaskGraphJson } from '@/types/task_graph_ir';
import { toast } from 'sonner';

interface FileWorkflowEditorProps {
  folderId: string;
  path: string;
  name: string;
  onClose: () => void;
  readOnly?: boolean;
  onEdit?: () => void;
  onUnsavedChange?: (hasUnsaved: boolean) => void;
}

export const FileWorkflowEditor: React.FC<FileWorkflowEditorProps> = ({
  folderId,
  path,
  name,
  onClose,
  readOnly = false,
  onEdit,
  onUnsavedChange,
}) => {
  const [initialGraph, setInitialGraph] = useState<TaskGraphJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasUnsaved, setHasUnsaved] = useState(false);

  useEffect(() => {
    const fetchContent = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          API_ENDPOINTS.workspace.files.read(folderId, path)
        );
        
        if (!response.ok) {
          throw new Error('Failed to load workflow file');
        }
        
        const data = await response.json();
        try {
          const json = JSON.parse(data.content);
          setInitialGraph(json);
        } catch (e) {
          setError('Invalid JSON content');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [folderId, path]);

  const handleSave = async (graph: TaskGraphJson) => {
    try {
      const response = await fetch(API_ENDPOINTS.workspace.files.write, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          folder_id: folderId,
          path: path,
          content: JSON.stringify(graph, null, 2),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save workflow');
      }
      
      // Mark as saved and show success toast
      setHasUnsaved(false);
      onUnsavedChange?.(false);
      toast.success('Workflow saved successfully');
    } catch (err) {
      console.error('Error saving workflow:', err);
      toast.error('Failed to save workflow');
    }
  };

  const handleChange = () => {
    if (!hasUnsaved) {
      setHasUnsaved(true);
      onUnsavedChange?.(true);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-red-500">
        <FileJson className="h-12 w-12 mb-4" />
        <p className="font-medium">Error loading workflow</p>
        <p className="text-sm mt-2">{error}</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-background flex flex-col">
      {readOnly && onEdit && (
        <div className="border-b p-4 flex items-center justify-between bg-background">
          <div>
            <h2 className="text-lg font-semibold">{name}</h2>
            <p className="text-sm text-muted-foreground">Workflow Preview (Read-only)</p>
          </div>
          <Button
            onClick={onEdit}
          >
            Edit Workflow
          </Button>
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        <WorkflowEditor 
          initialGraph={initialGraph || undefined}
          onSave={readOnly ? undefined : handleSave}
          readOnly={readOnly}
          onChange={readOnly ? undefined : handleChange}
        />
      </div>
    </div>
  );
};
