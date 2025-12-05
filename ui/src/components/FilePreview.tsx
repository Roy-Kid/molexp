/**
 * FilePreview Component
 * 
 * Main component for previewing file contents in the workspace.
 * Uses the plugin registry to resolve the appropriate preview renderer
 * based on file extension. Falls back to CodeEditor for unsupported types.
 */

import React, { useState, useEffect, Component, type ReactNode } from 'react';
import { Loader, AlertCircle, Code, Eye } from 'lucide-react';
import { ExplorerHeader } from './ExplorerHeader';
import { CodeEditor } from './CodeEditor';
import { Button } from '@/components/ui/button';
import { API_ENDPOINTS } from '@/config/api';
import { getMonacoLanguage } from '@/utils/file-utils';
import { 
  filePreviewPluginRegistry, 
  type FilePreviewContentProps 
} from '@/lib/file-preview-plugins';

// Import plugins to ensure they're registered
import '@/lib/plugins/markdown-plugin';
import '@/lib/plugins/workflow-plugin';

// ============================================================================
// Types
// ============================================================================

type PreviewMode = 'plugin' | 'raw';

interface FilePreviewProps {
  folderId: string;
  path: string;
  name: string;
  onToggleDetails?: () => void;
  onEdit?: () => void;
}

// ============================================================================
// Error Boundary for Plugin Rendering
// ============================================================================

interface PluginErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
  onError?: (error: Error) => void;
}

interface PluginErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary to catch errors in plugin rendering.
 * Falls back to the default code editor view on error.
 */
class PluginErrorBoundary extends Component<PluginErrorBoundaryProps, PluginErrorBoundaryState> {
  constructor(props: PluginErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): PluginErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('[FilePreview] Plugin rendering error:', error, errorInfo);
    this.props.onError?.(error);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// ============================================================================
// Default Code Preview (Fallback)
// ============================================================================

interface CodePreviewProps extends FilePreviewContentProps {}

/**
 * Default fallback preview using Monaco CodeEditor.
 * Used when no plugin matches the file type.
 */
const CodePreview: React.FC<CodePreviewProps> = ({ content, name }) => {
  return (
    <div className="h-full w-full">
      <CodeEditor
        value={content}
        language={getMonacoLanguage(name)}
        readOnly={true}
      />
    </div>
  );
};

// ============================================================================
// Preview Mode Toggle
// ============================================================================

interface PreviewModeToggleProps {
  mode: PreviewMode;
  onModeChange: (mode: PreviewMode) => void;
  hasPlugin: boolean;
  pluginName?: string;
}

const PreviewModeToggle: React.FC<PreviewModeToggleProps> = ({
  mode,
  onModeChange,
  hasPlugin,
  pluginName,
}) => {
  if (!hasPlugin) return null;

  return (
    <div className="flex items-center gap-0.5 p-0.5 bg-muted rounded-md">
      <Button
        variant={mode === 'plugin' ? 'default' : 'ghost'}
        size="sm"
        className={`h-7 px-3 text-xs gap-1.5 rounded-sm transition-all ${
          mode === 'plugin' 
            ? 'shadow-sm' 
            : 'text-muted-foreground hover:text-foreground'
        }`}
        onClick={() => onModeChange('plugin')}
        title={pluginName || 'Preview'}
      >
        <Eye className="h-3.5 w-3.5" />
        Preview
      </Button>
      <Button
        variant={mode === 'raw' ? 'default' : 'ghost'}
        size="sm"
        className={`h-7 px-3 text-xs gap-1.5 rounded-sm transition-all ${
          mode === 'raw' 
            ? 'shadow-sm' 
            : 'text-muted-foreground hover:text-foreground'
        }`}
        onClick={() => onModeChange('raw')}
        title="View raw text"
      >
        <Code className="h-3.5 w-3.5" />
        Raw
      </Button>
    </div>
  );
};

// ============================================================================
// Main FilePreview Component
// ============================================================================

export const FilePreview: React.FC<FilePreviewProps> = ({
  folderId,
  path,
  name,
}) => {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pluginError, setPluginError] = useState<Error | null>(null);
  const [previewMode, setPreviewMode] = useState<PreviewMode>('plugin');

  // Get the plugin for this file type
  const plugin = filePreviewPluginRegistry.getPluginForFile(name, path);

  useEffect(() => {
    fetchContent();
    // Reset to plugin mode when file changes
    setPreviewMode('plugin');
    setPluginError(null);
  }, [folderId, path]);

  const fetchContent = async () => {
    try {
      setLoading(true);
      setError(null);
      setPluginError(null);
      
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

  /**
   * Render the file content using the appropriate plugin or fallback.
   */
  const renderContent = () => {
    if (content === null) return null;

    const previewProps: FilePreviewContentProps = {
      content,
      name,
      path,
      folderId,
    };

    // User chose raw mode, show code editor
    if (previewMode === 'raw') {
      return <CodePreview {...previewProps} />;
    }

    // If we previously had a plugin error, use fallback
    if (pluginError) {
      return (
        <div className="h-full w-full flex flex-col">
          <div className="bg-yellow-500/10 border-b border-yellow-500/20 px-4 py-2 text-sm text-yellow-600 dark:text-yellow-400">
            Preview plugin failed. Showing raw content instead.
          </div>
          <div className="flex-1">
            <CodePreview {...previewProps} />
          </div>
        </div>
      );
    }

    if (plugin) {
      const PluginComponent = plugin.Component;
      return (
        <PluginErrorBoundary
          fallback={<CodePreview {...previewProps} />}
          onError={setPluginError}
        >
          <PluginComponent {...previewProps} />
        </PluginErrorBoundary>
      );
    }

    // No plugin found, use default code preview
    return <CodePreview {...previewProps} />;
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header with mode toggle */}
      <div className="flex items-center justify-between border-b px-4 py-2">
        <ExplorerHeader
          name={name}
          path={path}
          type="file"
        />
        <PreviewModeToggle
          mode={previewMode}
          onModeChange={setPreviewMode}
          hasPlugin={!!plugin}
          pluginName={plugin?.name}
        />
      </div>

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
            {renderContent()}
          </div>
        )}
      </div>
    </div>
  );
};
