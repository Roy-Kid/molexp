import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, File, Play, CheckCircle, XCircle, Clock, Loader, FolderPlus, X, Plus, FlaskConical, PlayCircle, FilePlus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { API_ENDPOINTS } from '@/config/api';
import { AddWorkspaceFolderDialog } from './AddWorkspaceFolderDialog';
import { Button } from './ui/button';

// Types
interface TreeNode {
  id: string;
  name: string;
  type: 'workspace' | 'project' | 'experiment' | 'run' | 'asset' | 'folder' | 'file';
  
  // Indexed folder metadata
  indexed?: boolean;           // Is this a molexp-managed indexed folder?
  kind?: string;               // Entity kind from backend (project, experiment, run, asset)
  schema_version?: string;     // Schema version for migrations
  
  // Existing fields
  children?: TreeNode[];
  status?: string;
  created?: string;
  parameters?: Record<string, any>;
  path?: string;
  size?: number;
  [key: string]: any;
}

interface WorkspaceFolder {
  id: string;
  name: string;
  path: string;
  added_at: string;
}

interface WorkspaceExplorerProps {
  onSelect?: (node: TreeNode) => void;
  onDoubleClick?: (node: TreeNode) => void;
  initialSelectedId?: string;
  onCreateProject?: () => void;
  onCreateExperiment?: (projectNode: TreeNode) => void;
  onCreateWorkflow?: (experimentNode: TreeNode) => void;
  onCreateFolder?: (parentNode: TreeNode) => void;
  onCreateFile?: (parentNode: TreeNode) => void;
  selectedNode?: TreeNode | null;
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

// File/Folder tree item component
const FileTreeItem: React.FC<{
  node: TreeNode;
  level: number;
  onSelect?: (node: TreeNode) => void;
  selectedId?: string;
  folderId: string;
  onLoadChildren: (folderId: string, path: string) => Promise<TreeNode[]>;
  onDoubleClick?: (node: TreeNode) => void;
}> = ({ node, level, onSelect, selectedId, folderId, onLoadChildren, onDoubleClick }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [children, setChildren] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(false);
  const isDirectory = node.type === 'folder';
  
  const handleClick = async () => {
    // Always select the node to show details
    onSelect?.(node);
  };
  
  const handleToggle = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!isDirectory) return;
    
    if (!isExpanded && children.length === 0) {
      // Load children
      setLoading(true);
      try {
        const loadedChildren = await onLoadChildren(folderId, node.path || '');
        setChildren(loadedChildren);
      } catch (err) {
        console.error('Failed to load children:', err);
      } finally {
        setLoading(false);
      }
    }
    setIsExpanded(!isExpanded);
  };
  
  const getIcon = () => {
    // Use kind if this is an indexed folder
    if (node.indexed && node.kind) {
      switch (node.kind) {
        case 'project':
          return isExpanded ? <FolderOpen className="h-4 w-4 text-blue-500" /> : <Folder className="h-4 w-4 text-blue-500" />;
        case 'experiment':
          return <FlaskConical className="h-4 w-4 text-purple-500" />;
        case 'run':
          return <Play className="h-4 w-4 text-green-500" />;
        case 'asset':
          return <File className="h-4 w-4 text-orange-500" />;
      }
    }
    
    // Regular folder/file icons
    if (isDirectory) {
      return isExpanded ? <FolderOpen className="h-4 w-4 text-yellow-500" /> : <Folder className="h-4 w-4 text-yellow-500" />;
    }
    return <File className="h-4 w-4 text-gray-500" />;
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
        onDoubleClick={() => onDoubleClick?.(node)}
      >
        {isDirectory && (
          <div className="flex-shrink-0" onClick={handleToggle}>
            {loading ? (
              <Loader className="h-4 w-4 text-muted-foreground animate-spin" />
            ) : isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        )}
        {!isDirectory && <div className="w-4" />}
        
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        
        <span className="text-sm truncate flex-1">
          {node.name}
        </span>
        
        {node.size !== undefined && (
          <span className="text-xs text-muted-foreground flex-shrink-0">
            {formatFileSize(node.size)}
          </span>
        )}
      </div>
      
      {isDirectory && isExpanded && children.length > 0 && (
        <div>
          {children.map((child) => (
            <FileTreeItem
              key={child.id}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedId={selectedId}
              folderId={folderId}
              onLoadChildren={onLoadChildren}
              onDoubleClick={onDoubleClick}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Tree item component for projects/experiments/runs
const TreeItem: React.FC<{
  node: TreeNode;
  level: number;
  onSelect?: (node: TreeNode) => void;
  selectedId?: string;
  onCreateExperiment?: (node: TreeNode) => void;
  onCreateWorkflow?: (node: TreeNode) => void;
}> = ({ node, level, onSelect, selectedId, onCreateExperiment, onCreateWorkflow }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2); // Auto-expand first 2 levels
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const hasChildren = node.children && node.children.length > 0;
  
  const handleClick = () => {
    // Always select the node to show details
    onSelect?.(node);
  };
  
  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!hasChildren) return;
    setIsExpanded(!isExpanded);
  };
  
  const handleContextMenu = (e: React.MouseEvent) => {
    // Only show context menu for projects and experiments
    if (node.type === 'project' || node.type === 'experiment') {
      e.preventDefault();
      e.stopPropagation();
      setContextMenu({ x: e.clientX, y: e.clientY });
    }
  };
  
  // Close context menu when clicking elsewhere
  React.useEffect(() => {
    if (contextMenu) {
      const closeMenu = () => setContextMenu(null);
      document.addEventListener('click', closeMenu);
      return () => document.removeEventListener('click', closeMenu);
    }
  }, [contextMenu]);
  
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
        onContextMenu={handleContextMenu}
      >
        {hasChildren && (
          <div className="flex-shrink-0" onClick={handleToggle}>
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
      
      {/* Context Menu */}
      {contextMenu && (
        <div
          className="fixed bg-popover border rounded-md shadow-md p-1 z-50 min-w-[160px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {node.type === 'project' && onCreateExperiment && (
            <div
              className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded-sm flex items-center gap-2"
              onClick={() => {
                onCreateExperiment(node);
                setContextMenu(null);
              }}
            >
              <FlaskConical className="h-4 w-4" />
              New Experiment
            </div>
          )}
          {node.type === 'experiment' && onCreateWorkflow && (
            <div
              className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded-sm flex items-center gap-2"
              onClick={() => {
                onCreateWorkflow(node);
                setContextMenu(null);
              }}
            >
              <PlayCircle className="h-4 w-4" />
              New Workflow
            </div>
          )}
        </div>
      )}
      
      {hasChildren && isExpanded && (
        <div>
          {node.children!.map((child) => (
            <TreeItem
              key={child.id}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedId={selectedId}
              onCreateExperiment={onCreateExperiment}
              onCreateWorkflow={onCreateWorkflow}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Helper function to format file size
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 10) / 10 + ' ' + sizes[i];
}

// Main component
export const WorkspaceExplorer: React.FC<WorkspaceExplorerProps> = ({ 
  onSelect, 
  onDoubleClick, 
  initialSelectedId, 
  onCreateProject,
  onCreateExperiment,
  onCreateWorkflow,
  onCreateFolder,
  onCreateFile,
  selectedNode,
}) => {
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [folders, setFolders] = useState<WorkspaceFolder[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | undefined>(initialSelectedId);
  const [dialogOpen, setDialogOpen] = useState(false);
  
  useEffect(() => {
    fetchTree();
    fetchFolders();
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

  const fetchFolders = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.workspace.folders.list);
      if (!response.ok) {
        throw new Error(`Failed to fetch folders: ${response.statusText}`);
      }
      
      const data = await response.json();
      setFolders(data);
    } catch (err) {
      console.error('Error fetching folders:', err);
    }
  };

  const loadFolderContents = React.useCallback(async (folderId: string, path: string): Promise<TreeNode[]> => {
    const response = await fetch(API_ENDPOINTS.workspace.folders.browse(folderId, path));
    if (!response.ok) {
      throw new Error(`Failed to browse folder: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Simply map entries without classification - indexed metadata comes from /api/workspace/tree
    return data.entries.map((entry: any) => ({
      id: `${folderId}:${entry.path}`,
      name: entry.name,
      type: entry.type === 'directory' ? 'folder' : 'file',
      path: entry.path,
      size: entry.size,
    }));
  }, []);  // Empty deps - function doesn't depend on any props/state

  const removeFolder = async (folderId: string) => {
    try {
      const response = await fetch(API_ENDPOINTS.workspace.folders.remove(folderId), {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to remove folder');
      }
      
      fetchFolders();
    } catch (err) {
      console.error('Error removing folder:', err);
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
  
  return (
    <div className="h-full overflow-auto">
      <div className="p-2">
        <div className="flex items-center justify-between mb-2 px-2">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground">
            Explorer
          </h2>
          <div className="flex items-center gap-1">
            {/* Context-sensitive New button */}
            {!selectedNode || selectedNode.type === 'workspace' ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={onCreateProject}
                title="New Project"
              >
                <Plus className="h-4 w-4" />
              </Button>
            ) : (selectedNode.indexed && selectedNode.kind === 'project') || selectedNode.type === 'project' ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={() => onCreateExperiment?.(selectedNode)}
                title="New Experiment"
              >
                <FlaskConical className="h-4 w-4" />
              </Button>
            ) : (selectedNode.indexed && selectedNode.kind === 'experiment') || selectedNode.type === 'experiment' ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={() => onCreateWorkflow?.(selectedNode)}
                title="New Workflow"
              >
                <PlayCircle className="h-4 w-4" />
              </Button>
            ) : selectedNode.type === 'folder' && !selectedNode.indexed ? (
              // Only show folder/file buttons for regular folders (not indexed entities)
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => onCreateFolder?.(selectedNode)}
                  title="New Folder"
                >
                  <FolderPlus className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => onCreateFile?.(selectedNode)}
                  title="New File"
                >
                  <FilePlus className="h-4 w-4" />
                </Button>
              </>
            ) : (
              // For files or indexed entities (runs, assets), show default New Project button
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={onCreateProject}
                title="New Project"
              >
                <Plus className="h-4 w-4" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={() => setDialogOpen(true)}
              title="Add Folder to Workspace"
            >
              <FolderPlus className="h-4 w-4" />
            </Button>
            <button
              onClick={() => {
                fetchTree();
                fetchFolders();
              }}
              className="text-xs text-muted-foreground hover:text-foreground"
              title="Refresh"
            >
              ↻
            </button>
          </div>
        </div>
        
        {/* Workspace Folders Section */}
        {folders.length > 0 && (
          <div className="mb-4">
            <div className="text-xs font-semibold text-muted-foreground px-2 mb-1">
              FOLDERS
            </div>
            {folders.map((folder) => (
              <div key={folder.id} className="mb-2">
                <div className="flex items-center gap-1 px-2 py-1 hover:bg-accent rounded-sm group">
                  <Folder className="h-4 w-4 text-blue-500 flex-shrink-0" />
                  <span className="text-sm truncate flex-1">{folder.name}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 opacity-0 group-hover:opacity-100"
                    onClick={() => removeFolder(folder.id)}
                    title="Remove folder"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
                <FolderContents
                  folderId={folder.id}
                  onSelect={handleSelect}
                  selectedId={selectedId}
                  onLoadChildren={loadFolderContents}
                  onDoubleClick={onDoubleClick}
                />
              </div>
            ))}
          </div>
        )}
        
        {/* Projects Section */}
        {tree && (
          <>
            <div className="text-xs font-semibold text-muted-foreground px-2 mb-1">
              PROJECTS
            </div>
            {tree.children && tree.children.length > 0 ? (
              tree.children.map((child) => (
                <TreeItem
                  key={child.id}
                  node={child}
                  level={0}
                  onSelect={handleSelect}
                  selectedId={selectedId}
                  onCreateExperiment={onCreateExperiment}
                  onCreateWorkflow={onCreateWorkflow}
                />
              ))
            ) : (
              <div className="px-2 py-4 text-sm text-muted-foreground">
                No projects yet. Create one to get started.
              </div>
            )}
          </>
        )}
      </div>
      
      <AddWorkspaceFolderDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onFolderAdded={fetchFolders}
      />
    </div>
  );
};

// Component to render folder contents
const FolderContents: React.FC<{
  folderId: string;
  onSelect?: (node: TreeNode) => void;
  selectedId?: string;
  onLoadChildren: (folderId: string, path: string) => Promise<TreeNode[]>;
  onDoubleClick?: (node: TreeNode) => void;
}> = ({ folderId, onSelect, selectedId, onLoadChildren, onDoubleClick }) => {
  const [children, setChildren] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadContents = async () => {
      setLoading(true);
      try {
        const contents = await onLoadChildren(folderId, '');
        setChildren(contents);
      } catch (err) {
        console.error('Failed to load folder contents:', err);
      } finally {
        setLoading(false);
      }
    };
    loadContents();
  }, [folderId, onLoadChildren]);

  if (loading) {
    return (
      <div className="px-2 py-1">
        <Loader className="h-4 w-4 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div>
      {children.map((child) => (
        <FileTreeItem
          key={child.id}
          node={child}
          level={1}
          onSelect={onSelect}
          selectedId={selectedId}
          folderId={folderId}
          onLoadChildren={onLoadChildren}
          onDoubleClick={onDoubleClick}
        />
      ))}
    </div>
  );
};
