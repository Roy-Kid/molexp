import React, { useEffect, useState } from 'react';
import { Button } from './ui/button';
import { Loader, FileJson, Edit, Activity, GitBranch } from 'lucide-react';
import { API_ENDPOINTS } from '@/config/api';

interface WorkflowPreviewProps {
  folderId: string;
  path: string;
  name: string;
  onEdit: () => void;
}

export const WorkflowPreview: React.FC<WorkflowPreviewProps> = ({
  folderId,
  path,
  name,
  onEdit,
}) => {
  const [content, setContent] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
          setContent(json);
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

  const nodeCount = content?.nodes?.length || 0;
  const edgeCount = content?.edges?.length || 0;
  const description = content?.description || 'No description provided';

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b p-6 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Activity className="h-6 w-6 text-blue-500" />
            <h1 className="text-2xl font-bold">{name}</h1>
          </div>
          <p className="text-muted-foreground">{path}</p>
        </div>
        <Button onClick={onEdit} className="gap-2">
          <Edit className="h-4 w-4" />
          Edit Workflow
        </Button>
      </div>

      {/* Content */}
      <div className="p-8 max-w-3xl">
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="p-6 rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Activity className="h-4 w-4" />
              <span className="text-sm font-medium">Nodes</span>
            </div>
            <div className="text-3xl font-bold">{nodeCount}</div>
          </div>
          <div className="p-6 rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <GitBranch className="h-4 w-4" />
              <span className="text-sm font-medium">Connections</span>
            </div>
            <div className="text-3xl font-bold">{edgeCount}</div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Description</h3>
          <p className="text-muted-foreground leading-relaxed">
            {description}
          </p>
        </div>

        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4">Preview</h3>
          <div className="rounded-md border bg-muted/50 p-4 font-mono text-xs overflow-auto max-h-96">
            <pre>{JSON.stringify(content, null, 2)}</pre>
          </div>
        </div>
      </div>
    </div>
  );
};
