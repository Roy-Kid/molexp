import React, { useState, useEffect } from 'react';
import { Loader, AlertCircle } from 'lucide-react';
import { ExplorerHeader } from './ExplorerHeader';
import { CodeEditor } from './CodeEditor';
import { API_ENDPOINTS } from '@/config/api';

interface FilePreviewProps {
  folderId: string;
  path: string;
  name: string;
  onToggleDetails: () => void;
}

export const FilePreview: React.FC<FilePreviewProps> = ({
  folderId,
  path,
  name,
}) => {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchContent();
  }, [folderId, path]);

  const fetchContent = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(API_ENDPOINTS.workspace.files.read(folderId, path));
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to load file content');
      }
      
      const data = await response.json();
      setContent(data.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getLanguage = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'py':
      case 'ipynb':
        return 'python';
      case 'ts':
      case 'tsx':
        return 'typescript';
      case 'js':
      case 'jsx':
        return 'javascript';
      case 'json':
        return 'json';
      case 'yml':
      case 'yaml':
        return 'yaml';
      case 'md':
        return 'markdown';
      case 'css':
        return 'css';
      case 'html':
        return 'html';
      default:
        return 'plaintext';
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      <ExplorerHeader
        name={name}
        path={path}
        type="file"
      />

      <div className="flex-1 overflow-hidden relative">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <AlertCircle className="h-8 w-8 text-red-500 mb-2" />
            <p className="text-sm font-medium text-red-500">{error}</p>
            <p className="text-xs text-muted-foreground mt-1">
              This file type might not be supported for preview.
            </p>
          </div>
        ) : (
          <div className="h-full w-full">
            <CodeEditor
              value={content || ''}
              language={getLanguage(name)}
              readOnly={true}
            />
          </div>
        )}
      </div>
    </div>
  );
};

