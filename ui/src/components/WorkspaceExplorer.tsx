import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, File, Play, CheckCircle, XCircle, Clock, Loader } from 'lucide-react';
import { cn } from '@/lib/utils';
import { API_ENDPOINTS } from '@/config/api';

// Types
interface TreeNode {
  id: string;
  name: string;
  type: 'workspace' | 'project' | 'experiment' | 'run' | 'asset';
  children?: TreeNode[];
  status?: string;
  created?: string;
  parameters?: Record<string, any>;
  [key: string]: any;
}

interface WorkspaceExplorerProps {
  onSelect?: (node: TreeNode) => void;
}

// Status icon component
const StatusIcon: React.FC<{ status?: string }> = ({ status }) => {
  if (!status) return null;
  
  switch (status.toLowerCase()) {
    case 'succeeded':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'failed':
      return <XCircle className="h-4 w-4 text-red-500" />;
    case 'running':
      return <Loader className="h-4 w-4 text-blue-500 animate-spin" />;
    case 'pending':
      return <Clock className="h-4 w-4 text-gray-400" />;
    default:
      return null;
  }
};

// Tree item component
const TreeItem: React.FC<{
  node: TreeNode;
  level: number;
  onSelect?: (node: TreeNode) => void;
  selectedId?: string;
}> = ({ node, level, onSelect, selectedId }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2); // Auto-expand first 2 levels
  const hasChildren = node.children && node.children.length > 0;
  
  const handleClick = () => {
    if (hasChildren) {
      setIsExpanded(!isExpanded);
    }
    onSelect?.(node);
  };
  
  const getIcon = () => {
    if (node.type === 'run') {
      return <Play className="h-4 w-4 text-blue-500" />;
    }
    if (node.type === 'experiment') {
      return <File className="h-4 w-4 text-purple-500" />;
    }
    if (node.type === 'project') {
      return isExpanded ? <FolderOpen className="h-4 w-4 text-yellow-500" /> : <Folder className="h-4 w-4 text-yellow-500" />;
    }
    return isExpanded ? <FolderOpen className="h-4 w-4" /> : <Folder className="h-4 w-4" />;
  };
  
  const isSelected = selectedId === node.id;
  
  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1 px-2 py-1 cursor-pointer hover:bg-accent rounded-sm",
          isSelected && "bg-accent"
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
      >
        {hasChildren && (
          <div className="flex-shrink-0">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        )}
        {!hasChildren && <div className="w-4" />}
        
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        
        <span className={cn(
          "text-sm truncate flex-1",
          node.type === 'run' && "font-mono text-xs"
        )}>
          {node.name}
        </span>
        
        {node.status && (
          <div className="flex-shrink-0">
            <StatusIcon status={node.status} />
          </div>
        )}
        
        {node.type === 'experiment' && node.runCount !== undefined && (
          <span className="text-xs text-muted-foreground flex-shrink-0">
            {node.runCount}
          </span>
        )}
      </div>
      
      {hasChildren && isExpanded && (
        <div>
          {node.children!.map((child) => (
            <TreeItem
              key={child.id}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedId={selectedId}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Main component
export const WorkspaceExplorer: React.FC<WorkspaceExplorerProps> = ({ onSelect }) => {
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  
  useEffect(() => {
    fetchTree();
  }, []);
  
  const fetchTree = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(API_ENDPOINTS.workspace.tree);
      if (!response.ok) {
        throw new Error(`Failed to fetch workspace tree: ${response.statusText}`);
      }
      
      const data = await response.json();
      setTree(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching workspace tree:', err);
    } finally {
      setLoading(false);
    }
  };

  
  const handleSelect = (node: TreeNode) => {
    setSelectedId(node.id);
    onSelect?.(node);
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-4 text-sm text-red-500">
        <p>Error loading workspace:</p>
        <p className="text-xs mt-1">{error}</p>
        <button
          onClick={fetchTree}
          className="mt-2 text-xs underline hover:no-underline"
        >
          Retry
        </button>
      </div>
    );
  }
  
  if (!tree) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        No workspace data
      </div>
    );
  }
  
  return (
    <div className="h-full overflow-auto">
      <div className="p-2">
        <div className="flex items-center justify-between mb-2 px-2">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground">
            Explorer
          </h2>
          <button
            onClick={fetchTree}
            className="text-xs text-muted-foreground hover:text-foreground"
            title="Refresh"
          >
            ↻
          </button>
        </div>
        
        {tree.children && tree.children.length > 0 ? (
          tree.children.map((child) => (
            <TreeItem
              key={child.id}
              node={child}
              level={0}
              onSelect={handleSelect}
              selectedId={selectedId}
            />
          ))
        ) : (
          <div className="px-2 py-4 text-sm text-muted-foreground">
            No projects yet. Create one to get started.
          </div>
        )}
      </div>
    </div>
  );
};
