import { ChevronRight } from "lucide-react";
import type { ComponentType, JSX, ReactNode } from "react";
import { useEffect, useState } from "react";
import { EMPTY_COPY, EmptyState } from "@/app/components/entity";

export interface TreeNode {
  id: string;
  label: string;
  icon?: ComponentType<{ className?: string }>;
  iconClassName?: string;
  right?: ReactNode;
  meta?: ReactNode;
  children?: TreeNode[];
  emptyChildLabel?: string;
  onSelect?: () => void;
}

interface TreeViewProps {
  nodes: TreeNode[];
  activeId?: string;
  expandPath?: string[];
  emptyTitle?: string;
  emptyDescription?: string;
  emptyIcon?: ReactNode;
}

const INDENT = 14;

interface RowProps {
  node: TreeNode;
  depth: number;
  activeId?: string;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}

const TreeRow = ({ node, depth, activeId, expanded, onToggle }: RowProps): JSX.Element => {
  const hasChildren = node.children !== undefined;
  const isExpanded = expanded.has(node.id);
  const isActive = activeId === node.id;
  const Icon = node.icon;

  return (
    <div>
      <div className="flex items-center gap-0.5" style={{ paddingLeft: `${depth * INDENT}px` }}>
        {hasChildren ? (
          <button
            type="button"
            aria-label={isExpanded ? "Collapse" : "Expand"}
            className="flex h-6 w-6 flex-none items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground"
            onClick={(event) => {
              event.stopPropagation();
              onToggle(node.id);
            }}
          >
            <ChevronRight
              className={`h-3.5 w-3.5 transition-transform ${isExpanded ? "rotate-90" : ""}`}
            />
          </button>
        ) : (
          <span className="h-6 w-6 flex-none" />
        )}
        <button
          type="button"
          className={`group flex h-7 min-w-0 flex-1 items-center gap-2 overflow-hidden rounded-md px-2 text-left text-sm transition-colors ${
            isActive ? "bg-accent text-accent-foreground" : "hover:bg-muted/40"
          }`}
          onClick={() => {
            if (node.onSelect) {
              node.onSelect();
            } else if (hasChildren) {
              onToggle(node.id);
            }
          }}
          title={node.label}
        >
          {Icon && (
            <Icon
              className={`h-3.5 w-3.5 flex-none ${node.iconClassName ?? "text-muted-foreground"}`}
            />
          )}
          <span className="min-w-0 flex-1 truncate">{node.label}</span>
          {node.right && <span className="flex-none">{node.right}</span>}
          {node.meta !== undefined && node.meta !== null && (
            <span className="flex-none font-mono text-[10px] text-muted-foreground">
              {node.meta}
            </span>
          )}
        </button>
      </div>
      {node.children !== undefined && isExpanded && (
        <div>
          {node.children.length === 0 && node.emptyChildLabel ? (
            <p
              className="text-xs text-muted-foreground"
              style={{ paddingLeft: `${(depth + 1) * INDENT + 8}px` }}
            >
              {node.emptyChildLabel}
            </p>
          ) : (
            node.children.map((child) => (
              <TreeRow
                key={child.id}
                node={child}
                depth={depth + 1}
                activeId={activeId}
                expanded={expanded}
                onToggle={onToggle}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
};

export const TreeView = ({
  nodes,
  activeId,
  expandPath,
  emptyTitle,
  emptyDescription,
  emptyIcon,
}: TreeViewProps): JSX.Element => {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(expandPath ?? []));

  useEffect(() => {
    if (!expandPath || expandPath.length === 0) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      for (const id of expandPath) next.add(id);
      return next;
    });
  }, [expandPath]);

  const toggle = (id: string): void => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (nodes.length === 0) {
    return (
      <EmptyState
        title={emptyTitle ?? EMPTY_COPY.entries.title}
        description={emptyDescription}
        icon={emptyIcon}
        density="compact"
      />
    );
  }

  return (
    <div className="space-y-0.5">
      {nodes.map((node) => (
        <TreeRow
          key={node.id}
          node={node}
          depth={0}
          activeId={activeId}
          expanded={expanded}
          onToggle={toggle}
        />
      ))}
    </div>
  );
};
