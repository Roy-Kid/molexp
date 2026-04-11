/**
 * Tree component based on Radix UI patterns
 * Provides a low-level, file-system-oriented tree view
 * No semantic meaning imposed - purely structural hierarchy
 */

import { ChevronRight, File, Folder, FolderOpen } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/utils";

interface TreeNodeProps {
  id: string;
  name: string;
  path: string;
  kind: "file" | "folder";
  children?: TreeNodeProps[];
  icon?: React.ReactNode;
  metadata?: Record<string, unknown>;
}

interface TreeProps {
  nodes: TreeNodeProps[];
  onSelect?: (node: TreeNodeProps) => void;
  onContextMenu?: (node: TreeNodeProps, event: React.MouseEvent) => void;
  defaultExpandedIds?: Set<string>;
  className?: string;
  renderNode?: (node: TreeNodeProps, defaultRender: React.ReactNode) => React.ReactNode;
}

interface TreeItemProps {
  node: TreeNodeProps;
  level: number;
  isExpanded: boolean;
  onToggle: (id: string) => void;
  onSelect?: (node: TreeNodeProps) => void;
  onContextMenu?: (node: TreeNodeProps, event: React.MouseEvent) => void;
  renderNode?: (node: TreeNodeProps, defaultRender: React.ReactNode) => React.ReactNode;
}

interface TreeItemRendererProps {
  node: TreeNodeProps;
  level: number;
  expandedIds?: Map<string, boolean>;
  onToggle: (id: string) => void;
  onSelect?: (node: TreeNodeProps) => void;
  onContextMenu?: (node: TreeNodeProps, event: React.MouseEvent) => void;
  renderNode?: (node: TreeNodeProps, defaultRender: React.ReactNode) => React.ReactNode;
}

/**
 * TreeItem - Individual tree node renderer
 * Follows VS Code explorer styling and interactions
 */
const TreeItem = React.forwardRef<HTMLButtonElement, TreeItemProps>(
  ({ node, level, isExpanded, onToggle, onSelect, onContextMenu, renderNode }, ref) => {
    const hasChildren = node.children && node.children.length > 0;
    const isFolder = node.kind === "folder";

    const handleToggle = (e: React.MouseEvent) => {
      e.stopPropagation();
      if (hasChildren) {
        onToggle(node.id);
      }
    };

    const handleSelect = () => {
      onSelect?.(node);
    };

    const handleContextMenu = (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      onContextMenu?.(node, e);
    };

    const DefaultRender = (
      <button
        type="button"
        ref={ref}
        className="group flex items-center min-h-[28px] w-full px-2 py-1 rounded-sm cursor-pointer select-none hover:bg-accent/50 transition-colors text-left"
        style={{ paddingLeft: `${level * 12 + 4}px` }}
        onClick={handleSelect}
        onContextMenu={handleContextMenu}
        data-node-id={node.id}
        data-node-path={node.path}
      >
        {/* Expand/collapse toggle */}
        {isFolder && (
          <button
            type="button"
            className={cn(
              "flex items-center justify-center w-5 h-5 mr-1 flex-shrink-0",
              "text-muted-foreground hover:text-foreground transition-colors",
              hasChildren ? "cursor-pointer" : "cursor-default opacity-0",
            )}
            onClick={handleToggle}
            aria-label={isExpanded ? "Collapse" : "Expand"}
            disabled={!hasChildren}
          >
            <ChevronRight
              className={cn("h-4 w-4 transition-transform", isExpanded && "rotate-90")}
            />
          </button>
        )}

        {/* File/folder icon */}
        <div className="flex items-center justify-center w-5 h-5 mr-2 flex-shrink-0 text-muted-foreground">
          {node.icon ? (
            node.icon
          ) : isFolder ? (
            isExpanded ? (
              <FolderOpen className="h-4 w-4" />
            ) : (
              <Folder className="h-4 w-4" />
            )
          ) : (
            <File className="h-4 w-4" />
          )}
        </div>

        {/* Node label */}
        <span className="truncate text-sm text-foreground flex-1">{node.name}</span>
      </button>
    );

    return (
      <div className="flex flex-col">
        {renderNode ? renderNode(node, DefaultRender) : DefaultRender}
        {isFolder && isExpanded && hasChildren && (
          <div className="flex flex-col">
            {node.children?.map((child) => (
              <TreeItemRenderer
                key={child.id}
                node={child}
                level={level + 1}
                onToggle={onToggle}
                onSelect={onSelect}
                onContextMenu={onContextMenu}
                renderNode={renderNode}
              />
            ))}
          </div>
        )}
        {isFolder && isExpanded && !hasChildren && (
          <div
            className="text-xs text-muted-foreground italic px-2 py-1"
            style={{ paddingLeft: `${(level + 1) * 12 + 4}px` }}
          >
            Empty folder
          </div>
        )}
      </div>
    );
  },
);
TreeItem.displayName = "TreeItem";

/**
 * TreeItemRenderer - Manages expanded state for tree items
 */
const TreeItemRenderer = React.memo(
  ({ node, level, expandedIds, onToggle, ...props }: TreeItemRendererProps) => {
    const [expandedState, setExpandedState] = React.useState(
      expandedIds?.get(node.id) ?? level < 2,
    );

    const handleToggle = (id: string) => {
      setExpandedState(!expandedState);
      onToggle(id);
    };

    return (
      <TreeItem
        node={node}
        level={level}
        isExpanded={expandedState}
        onToggle={handleToggle}
        {...props}
      />
    );
  },
);
TreeItemRenderer.displayName = "TreeItemRenderer";

/**
 * Tree - Root tree component
 * Manages expansion state globally
 */
const Tree = React.forwardRef<HTMLDivElement, TreeProps>(
  ({ nodes, onSelect, onContextMenu, defaultExpandedIds, className, renderNode }, ref) => {
    const [expandedIds, setExpandedIds] = React.useState<Map<string, boolean>>(() => {
      const map = new Map<string, boolean>();
      if (defaultExpandedIds) {
        for (const id of defaultExpandedIds) {
          map.set(id, true);
        }
      }
      return map;
    });

    const handleToggle = React.useCallback((id: string) => {
      setExpandedIds((prev) => {
        const next = new Map(prev);
        next.set(id, !next.get(id));
        return next;
      });
    }, []);

    if (nodes.length === 0) {
      return (
        <div className={cn("text-sm text-muted-foreground italic p-4", className)}>
          No files or folders
        </div>
      );
    }

    return (
      <div ref={ref} className={cn("flex flex-col", className)}>
        {nodes.map((node) => (
          <TreeItemRenderer
            key={node.id}
            node={node}
            level={0}
            expandedIds={expandedIds}
            onToggle={handleToggle}
            onSelect={onSelect}
            onContextMenu={onContextMenu}
            renderNode={renderNode}
          />
        ))}
      </div>
    );
  },
);
Tree.displayName = "Tree";

export { Tree, TreeItem, type TreeNodeProps, type TreeProps };
