/**
 * FileExplorer - Workspace file explorer
 *
 * Low-level, file-oriented navigation surface
 * Presents workspace as hierarchical, collapsible file structure
 * Does NOT impose semantic meaning beyond file system structure
 *
 * Separated from semantic views (Project, Experiment, Run, Asset)
 */

import { Copy, RefreshCw, Trash2 } from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tree, type TreeNodeProps } from "@/components/ui/tree";
import { cn } from "@/lib/utils";

interface FileExplorerProps {
  onSelectFile?: (path: string, node: TreeNodeProps) => void;
  onSelectFolder?: (path: string, node: TreeNodeProps) => void;
  rootPath?: string;
  className?: string;
}

interface ExplorerContextMenuProps {
  node: TreeNodeProps;
  onCopy?: (path: string) => void;
  onDelete?: (path: string) => void;
  event: React.MouseEvent;
}

/**
 * ExplorerContextMenu - Right-click menu for file operations
 */
const ExplorerContextMenu = ({ node, onCopy, onDelete, event }: ExplorerContextMenuProps) => {
  const position = { x: event.clientX, y: event.clientY };

  return (
    <div
      className="fixed bg-popover border border-border rounded-md shadow-md z-50 min-w-[150px]"
      role="menu"
      style={{
        top: `${position.y}px`,
        left: `${position.x}px`,
      }}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === "Escape") e.stopPropagation();
      }}
    >
      <div className="py-1">
        <button
          type="button"
          className="w-full text-left px-2 py-1.5 text-sm hover:bg-accent flex items-center gap-2"
          onClick={() => {
            onCopy?.(node.path);
          }}
        >
          <Copy className="h-3.5 w-3.5" />
          Copy path
        </button>
        <button
          type="button"
          className="w-full text-left px-2 py-1.5 text-sm hover:bg-accent text-destructive flex items-center gap-2"
          onClick={() => {
            onDelete?.(node.path);
          }}
        >
          <Trash2 className="h-3.5 w-3.5" />
          Delete
        </button>
      </div>
    </div>
  );
};

const transformToTreeNodes = (items: unknown[]): TreeNodeProps[] => {
  return items.map((rawItem) => {
    const item = rawItem as Record<string, unknown>;
    return {
      id: (item.id as string) || (item.path as string),
      name: item.name as string,
      path: item.path as string,
      kind: item.type === "folder" ? "folder" : ("file" as "folder" | "file"),
      children: Array.isArray(item.children) ? transformToTreeNodes(item.children) : undefined,
      metadata: {
        size: item.size,
        modified: item.modified,
        type: item.type,
      },
    };
  });
};

/**
 * FileExplorer - Main component
 *
 * Provides pure file-system tree navigation
 * - No semantic interpretation of files
 * - No project/experiment/run context
 * - Pure hierarchical file structure
 */
export const FileExplorer = React.forwardRef<HTMLDivElement, FileExplorerProps>(
  ({ onSelectFile, onSelectFolder, rootPath = "/", className }, ref) => {
    const [treeNodes, setTreeNodes] = useState<TreeNodeProps[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [contextMenu, setContextMenu] = useState<{
      node: TreeNodeProps;
      event: React.MouseEvent;
    } | null>(null);

    const loadWorkspaceTree = useCallback(async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch the workspace tree from API
        // This should return the pure file-system structure
        const response = await fetch(`/api/workspace/files?path=${encodeURIComponent(rootPath)}`);

        if (!response.ok) {
          throw new Error(`Failed to load workspace: ${response.statusText}`);
        }

        const data = await response.json();
        setTreeNodes(transformToTreeNodes(data.children || []));
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
        console.error("Error loading workspace tree:", err);
      } finally {
        setLoading(false);
      }
    }, [rootPath]);

    // Load root workspace files
    useEffect(() => {
      loadWorkspaceTree();
    }, [loadWorkspaceTree]);

    const handleSelectNode = (node: TreeNodeProps) => {
      if (node.kind === "file") {
        onSelectFile?.(node.path, node);
      } else {
        onSelectFolder?.(node.path, node);
      }
    };

    const handleContextMenu = (node: TreeNodeProps, event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      setContextMenu({ node, event });
    };

    const handleCopyPath = (path: string) => {
      navigator.clipboard.writeText(path);
      setContextMenu(null);
    };

    const handleDeleteFile = (path: string) => {
      // TODO: Implement file deletion
      console.log("Delete:", path);
      setContextMenu(null);
    };

    if (error) {
      return (
        <div className={cn("flex flex-col gap-2 p-4", className)} ref={ref}>
          <div className="text-sm text-destructive">
            <p className="font-semibold">Error loading workspace</p>
            <p className="text-xs">{error}</p>
          </div>
          <Button variant="outline" size="sm" onClick={loadWorkspaceTree} className="w-full">
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Retry
          </Button>
        </div>
      );
    }

    return (
      <div ref={ref} className={cn("flex flex-col h-full bg-background", className)}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Files
          </h3>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={loadWorkspaceTree}
            disabled={loading}
            title="Refresh"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          </Button>
        </div>

        {/* Tree content */}
        <ScrollArea className="flex-1 px-2 py-2">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-xs text-muted-foreground">Loading...</div>
            </div>
          ) : treeNodes.length === 0 ? (
            <div className="text-xs text-muted-foreground p-4">No files or folders found</div>
          ) : (
            <Tree nodes={treeNodes} onSelect={handleSelectNode} onContextMenu={handleContextMenu} />
          )}
        </ScrollArea>

        {/* Context menu */}
        {contextMenu && (
          <ExplorerContextMenu
            node={contextMenu.node}
            event={contextMenu.event}
            onCopy={handleCopyPath}
            onDelete={handleDeleteFile}
          />
        )}
      </div>
    );
  },
);

FileExplorer.displayName = "FileExplorer";
