import React, { useState, useEffect } from 'react';
import { Save, X, Loader, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CodeEditor } from './CodeEditor';
import { API_ENDPOINTS } from '@/config/api';
import { cn } from '@/lib/utils';

interface FileEditorProps {
  folderId: string;
  path: string;
  name: string;
  onClose: () => void;
  onSaveSuccess?: () => void;
}

export const FileEditor: React.FC<FileEditorProps> = ({
  folderId,
  path,
  name,
  onClose,
  onSaveSuccess,
}) => {
  const [content, setContent] = useState<string>('');
  const [originalContent, setOriginalContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
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
        throw new Error('Failed to load file content');
      }
      
      const data = await response.json();
      setContent(data.content);
      setOriginalContent(data.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      
      const response = await fetch(API_ENDPOINTS.workspace.files.write, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          folder_id: folderId,
          path: path,
          content: content,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to save file');
      }
      
      setOriginalContent(content);
      onSaveSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setSaving(false);
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

  const hasChanges = content !== originalContent;

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-background">
        <div className="flex flex-col overflow-hidden">
          <h1 className="text-lg font-semibold truncate flex items-center gap-2">
            {name}
            {hasChanges && <span className="text-xs text-yellow-500 font-normal">(unsaved)</span>}
          </h1>
          <div className="text-xs text-muted-foreground truncate font-mono">
            {path}
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            disabled={saving}
          >
            <X className="h-4 w-4 mr-1" />
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={saving || !hasChanges}
            className={cn(hasChanges && "bg-blue-600 hover:bg-blue-700")}
          >
            {saving ? (
              <Loader className="h-4 w-4 animate-spin mr-1" />
            ) : (
              <Save className="h-4 w-4 mr-1" />
            )}
            Save
          </Button>
        </div>
      </div>

      {/* Editor Area */}
      <div className="flex-1 overflow-hidden relative">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <AlertCircle className="h-8 w-8 text-red-500 mb-2" />
            <p className="text-sm font-medium text-red-500">{error}</p>
            <Button variant="outline" size="sm" onClick={fetchContent} className="mt-4">
              Retry
            </Button>
          </div>
        ) : (
          <div className="h-full w-full">
            <CodeEditor
              value={content}
              language={getLanguage(name)}
              onChange={(val) => setContent(val || '')}
            />
          </div>
        )}
      </div>
    </div>
  );
};
