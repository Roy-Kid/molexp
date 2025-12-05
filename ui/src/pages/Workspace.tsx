import React, { useState } from 'react';
import { WorkspaceExplorer } from '@/components/WorkspaceExplorer';
import { DetailPanel } from '@/components/DetailPanel';
import { FilePreview } from '@/components/FilePreview';
import { FileEditor } from '@/components/FileEditor';
import { FileWorkflowEditor } from '@/components/FileWorkflowEditor';
import { DetailOverlay } from '@/components/DetailOverlay';
import { Button } from '@/components/ui/button';
import { CreateProjectDialog } from '@/components/CreateProjectDialog';
import { CreateExperimentDialog } from '@/components/CreateExperimentDialog';
import { CreateWorkflowDialog } from '@/components/CreateWorkflowDialog';
import { API_ENDPOINTS } from '@/config/api';
import { PanelRight } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

interface TreeNode {
  id: string;
  name: string;
  type: 'workspace' | 'project' | 'experiment' | 'run' | 'asset' | 'folder' | 'file';
  path?: string;
  [key: string]: any;
}

export const Workspace: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [isCreateExperimentOpen, setIsCreateExperimentOpen] = useState(false);
  const [isCreateWorkflowOpen, setIsCreateWorkflowOpen] = useState(false);
  const [explorerKey, setExplorerKey] = useState(0);
  const [autoSelectId, setAutoSelectId] = useState<string | undefined>(undefined);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [pendingNode, setPendingNode] = useState<TreeNode | null>(null);
  const [showUnsavedDialog, setShowUnsavedDialog] = useState(false);



  const handleCreateFolder = async () => {
    if (!selectedNode || (selectedNode.type !== 'folder' && selectedNode.type !== 'file')) return;
    
    const folderId = selectedNode.id.split(':')[0];
    // Create sibling: use parent path
    const currentPath = selectedNode.path || '';
    const parentPath = currentPath.split('/').slice(0, -1).join('/');
    
    const baseName = 'New Folder';
    let name = baseName;
    
    // Simple retry logic or just try once with timestamp if collision?
    // For now, let's try a timestamp to avoid collision easily without checking
    name = `${baseName} ${Date.now()}`;

    try {
        const response = await fetch(API_ENDPOINTS.workspace.files.createDirectory, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder_id: folderId,
                path: parentPath ? `${parentPath}/${name}` : name,
            }),
        });
        
        if (!response.ok) throw new Error('Failed to create folder');
        
        // Refresh
        setExplorerKey(k => k + 1);
    } catch (err) {
        console.error(err);
        alert('Failed to create folder');
    }
  };

  const handleCreateFile = async () => {
    if (!selectedNode || (selectedNode.type !== 'folder' && selectedNode.type !== 'file')) return;
    
    const folderId = selectedNode.id.split(':')[0];
    // Create sibling
    const currentPath = selectedNode.path || '';
    const parentPath = currentPath.split('/').slice(0, -1).join('/');
    
    const name = `Untitled ${Date.now()}.txt`;

    try {
        const response = await fetch(API_ENDPOINTS.workspace.files.write, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder_id: folderId,
                path: parentPath ? `${parentPath}/${name}` : name,
                content: '',
            }),
        });
        
        if (!response.ok) throw new Error('Failed to create file');
        
        // Refresh
        setExplorerKey(k => k + 1);
    } catch (err) {
        console.error(err);
        alert('Failed to create file');
    }
  };
  
  const handleSelect = (node: TreeNode) => {
    // If clicking the same workflow file while in edit mode, toggle to preview
    if (isEditing && node.id === selectedNode?.id && node.type === 'file' && node.path?.endsWith('.flow')) {
      if (hasUnsavedChanges) {
        // Show confirmation dialog if there are unsaved changes
        setPendingNode(node);
        setShowUnsavedDialog(true);
        return;
      }
      // Switch to preview mode
      setIsEditing(false);
      setHasUnsavedChanges(false);
      return;
    }
    
    // Only show warning if we have unsaved changes AND we're switching to a different file
    if (isEditing && hasUnsavedChanges && node.id !== selectedNode?.id) {
      setPendingNode(node);
      setShowUnsavedDialog(true);
      return;
    }
    
    // Only update if it's actually a different node
    if (node.id !== selectedNode?.id || node.type !== selectedNode?.type) {
      console.log('[Workspace] Selecting new node:', node.id, node.type);
      setSelectedNode(node);
    } else {
      console.log('[Workspace] Same node clicked, skipping update:', node.id);
    }
  };
  
  const handleConfirmDiscard = () => {
    setIsEditing(false);
    setHasUnsavedChanges(false);
    setShowUnsavedDialog(false);
    
    if (pendingNode) {
      // If it's the same node, just toggle to preview
      if (pendingNode.id === selectedNode?.id) {
        // Already handled by setting isEditing to false
      } else {
        // Switch to the pending node
        setSelectedNode(pendingNode);
      }
      setPendingNode(null);
    }
  };
  
  const handleCancelDiscard = () => {
    setShowUnsavedDialog(false);
    setPendingNode(null);
  };

  const handleDoubleClick = (node: TreeNode) => {
    if (node.type === 'file') {
      setIsEditing(true);
    }
  };
  
  const renderMainContent = () => {
    if (!selectedNode) {
      return (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          Select an item to view
        </div>
      );
    }

    if (selectedNode.type === 'folder') {
      return (
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">{selectedNode.name}</h1>
          <p className="text-muted-foreground mb-4">
            Workspace Folder
          </p>
          <p className="text-sm text-muted-foreground">
            Browse folder contents in the explorer on the left, or select "Show Details" to view folder information.
          </p>
        </div>
      );
    }

    if (selectedNode.type === 'file') {
      const folderId = selectedNode.id.split(':')[0];
      const path = selectedNode.path || '';
      
      // Check for workflow file in edit mode
      if (path.endsWith('.flow') && isEditing) {
        return (
          <FileWorkflowEditor
            key={`${folderId}:${path}:edit`}
            folderId={folderId}
            path={path}
            name={selectedNode.name}
            onClose={() => {
              setIsEditing(false);
              setHasUnsavedChanges(false);
            }}
            onUnsavedChange={(unsaved) => setHasUnsavedChanges(unsaved)}
          />
        );
      }
      
      // Regular file editing (non-workflow)
      if (isEditing) {
        return (
          <FileEditor
            key={`${folderId}:${path}:edit`}
            folderId={folderId}
            path={path}
            name={selectedNode.name}
            onClose={() => setIsEditing(false)}
            onSaveSuccess={() => {
              // Optional: show toast
            }}
          />
        );
      }
      
      // Preview mode - uses plugin system for all file types including .flow
      return (
        <FilePreview
          key={`${folderId}:${path}:preview`}
          folderId={folderId}
          path={path}
          name={selectedNode.name}
          onToggleDetails={() => setShowDetails(!showDetails)}
          onEdit={() => setIsEditing(true)}
        />
      );
    }

    // Handle Project and Experiment - show simple overview
    if (selectedNode.type === 'project') {
      return (
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">{selectedNode.name}</h1>
          <p className="text-muted-foreground mb-4">
            Project with {selectedNode.experimentCount || 0} experiments
          </p>
          <p className="text-sm text-muted-foreground">
            Select "Show Details" in the top right to view full project information.
          </p>
        </div>
      );
    }

    if (selectedNode.type === 'experiment') {
      return (
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">{selectedNode.name}</h1>
          <p className="text-muted-foreground mb-4">
            Experiment with {selectedNode.runCount || 0} runs
          </p>
          <p className="text-sm text-muted-foreground">
            Select "Show Details" in the top right to view full experiment information.
          </p>
        </div>
      );
    }

    // Fallback for other types
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">{selectedNode.name}</h1>
        <p className="text-muted-foreground">
          Select "Show Details" to view properties.
        </p>
      </div>
    );
  };
  
  // Memoize main content to prevent unnecessary re-renders when unrelated state changes
  // Use selectedNode.id instead of selectedNode to avoid re-renders when same node is re-selected
  const mainContent = React.useMemo(() => {
    console.log('[Workspace] Rendering main content for:', selectedNode?.type, selectedNode?.id);
    return renderMainContent();
  }, [selectedNode?.id, selectedNode?.type, isEditing, showDetails]);

  return (
    <div className="flex h-screen bg-background">
      {/* Left - Explorer */}
      <div className="w-64 border-r flex flex-col">
        <WorkspaceExplorer 
          key={explorerKey}
          onSelect={handleSelect} 
          onDoubleClick={handleDoubleClick}
          initialSelectedId={autoSelectId}
          selectedNode={selectedNode}
          onCreateProject={() => setIsCreateProjectOpen(true)}
          onCreateExperiment={(node) => {
            handleSelect(node);
            setIsCreateExperimentOpen(true);
          }}
          onCreateWorkflow={(node) => {
            handleSelect(node);
            setIsCreateWorkflowOpen(true);
          }}
          onCreateFolder={async (node) => {
            handleSelect(node);
            // Small delay to ensure selection state updates
            setTimeout(() => handleCreateFolder(), 0);
          }}
          onCreateFile={async (node) => {
            handleSelect(node);
            setTimeout(() => handleCreateFile(), 0);
          }}
        />
      </div>
      
      {/* Center - Main Content */}
      <div className="flex-1 bg-background overflow-hidden relative flex flex-col">
        {/* Header */}
        <div className="h-14 border-b flex items-center justify-between px-6 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="flex items-center gap-2 font-medium">
                {selectedNode ? (
                    <>
                        <span className="text-muted-foreground">{selectedNode.type.charAt(0).toUpperCase() + selectedNode.type.slice(1)}</span>
                        <span className="text-muted-foreground">/</span>
                        <span>{selectedNode.name}{hasUnsavedChanges && isEditing ? ' *' : ''}</span>
                    </>
                ) : (
                    <span>Workspace</span>
                )}
            </div>
            <div className="flex items-center gap-2">
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowDetails(!showDetails)}
                    title="Toggle Details"
                    className={showDetails ? "bg-accent" : ""}
                >
                    <PanelRight className="h-4 w-4" />
                </Button>
            </div>
        </div>

        <div className="flex-1 overflow-hidden relative">
            {mainContent}
        </div>
        
        {/* Right - Detail Overlay */}
        <DetailOverlay 
          isOpen={showDetails} 
          onClose={() => setShowDetails(false)}
        >
          <DetailPanel
            nodeId={selectedNode?.id || null}
            nodeType={selectedNode?.type === 'workspace' ? null : (selectedNode?.type as any) || null}
          />
        </DetailOverlay>

        <CreateProjectDialog 
          isOpen={isCreateProjectOpen} 
          onClose={() => setIsCreateProjectOpen(false)} 
          onSuccess={(newId) => {
             setAutoSelectId(newId);
             setExplorerKey(k => k + 1);
          }} 
        />
        <CreateExperimentDialog 
          isOpen={isCreateExperimentOpen} 
          onClose={() => setIsCreateExperimentOpen(false)} 
          onSuccess={(newId) => {
             setAutoSelectId(newId);
             setExplorerKey(k => k + 1);
          }} 
          projectId={selectedNode?.type === 'project' ? selectedNode.id : ''}
        />
        <CreateWorkflowDialog 
          isOpen={isCreateWorkflowOpen} 
          onClose={() => setIsCreateWorkflowOpen(false)} 
          onSuccess={(newId) => {
             // Parse the workflow file path from the returned ID
             // Format is "workspace:path/to/file.flow"
             const path = newId.replace('workspace:', '');
             
             // Create a proper node for the workflow file
             const workflowNode: TreeNode = {
               id: newId,
               name: path.split('/').pop() || 'workflow.flow',
               type: 'file',
               path: path,
             };
             
             // Select the workflow file
             setSelectedNode(workflowNode);
             setAutoSelectId(newId);
             
             // Open in edit mode
             setIsEditing(true);
             
             // Refresh explorer
             setExplorerKey(k => k + 1);
          }} 
          folderId="workspace"
          basePath={selectedNode?.type === 'experiment' ? `projects/${selectedNode.id.split('/')[0]}/experiments/${selectedNode.id.split('/')[1]}` : ''}
        />
      </div>
      
      {/* Unsaved Changes Alert Dialog */}
      <AlertDialog open={showUnsavedDialog} onOpenChange={setShowUnsavedDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Unsaved Changes</AlertDialogTitle>
            <AlertDialogDescription>
              You have unsaved changes. Are you sure you want to discard them?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleCancelDiscard}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmDiscard}>Discard Changes</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
