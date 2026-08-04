import { ChevronRight } from "lucide-react";
import type { ComponentType, JSX, ReactNode } from "react";
import { Fragment, useEffect, useState } from "react";
import { EMPTY_COPY, EmptyState } from "@/app/components/entity";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { WorkbenchAction, WorkbenchIconAction } from "@/components/workbench";

export interface TreeNodeAction {
  id: string;
  label: string;
  icon?: ComponentType<{ className?: string }>;
  disabled?: boolean;
  destructive?: boolean;
  separatorBefore?: boolean;
  title?: string;
  onSelect: () => void;
}

export interface TreeNode {
  id: string;
  label: string;
  /** Hover tooltip; defaults to `label` (e.g. an agent task's full goal). */
  hoverTitle?: string;
  labelClassName?: string;
  icon?: ComponentType<{ className?: string }>;
  iconClassName?: string;
  leadingAccessory?: ReactNode;
  right?: ReactNode;
  meta?: ReactNode;
  children?: TreeNode[];
  emptyChildLabel?: string;
  actions?: TreeNodeAction[];
  onSelect?: () => void;
}

interface TreeViewProps {
  nodes: TreeNode[];
  activeId?: string;
  expandPath?: string[];
  emptyTitle?: string;
  emptyDescription?: string;
  emptyIcon?: ReactNode;
  /** Fired when a node is expanded (not collapsed). Used for lazy WorkspaceFs.listdir. */
  onExpand?: (nodeId: string) => void;
}

const INDENT = 14;

interface RowProps {
  node: TreeNode;
  depth: number;
  /** True if any sibling at this level has children. Drives the chevron-column reservation. */
  reserveChevron: boolean;
  activeId?: string;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}

const TreeRow = ({
  node,
  depth,
  reserveChevron,
  activeId,
  expanded,
  onToggle,
}: RowProps): JSX.Element => {
  const hasChildren = node.children !== undefined;
  const isExpanded = expanded.has(node.id);
  const isActive = activeId === node.id;
  const Icon = node.icon;
  const actions = node.actions ?? [];
  const childrenReserveChevron = node.children?.some((c) => c.children !== undefined) ?? false;

  const rowButton = (
    <WorkbenchAction
      kind="ghost"
      size="content"
      type="button"
      className={`group flex h-control-compact min-w-0 flex-1 items-center gap-2 overflow-hidden rounded-control px-2 text-left text-body-lg transition-colors ${
        // Soft lavender wash — solid bg-accent is for buttons, too dark on day mode rows.
        isActive ? "bg-accent-muted text-accent-muted-foreground" : "hover:bg-muted/40"
      }`}
      onClick={() => {
        if (node.onSelect) {
          node.onSelect();
        } else if (hasChildren) {
          onToggle(node.id);
        }
      }}
      onContextMenu={() => {
        node.onSelect?.();
      }}
      title={node.hoverTitle ?? node.label}
    >
      {Icon && (
        <Icon
          className={`h-3.5 w-3.5 flex-none ${node.iconClassName ?? "text-muted-foreground"}`}
        />
      )}
      {node.leadingAccessory && <span className="flex-none">{node.leadingAccessory}</span>}
      <span className={`min-w-0 flex-1 truncate ${node.labelClassName ?? ""}`}>{node.label}</span>
      {node.right && <span className="flex-none">{node.right}</span>}
      {node.meta !== undefined && node.meta !== null && (
        <span className="flex-none font-mono text-micro text-muted-foreground">{node.meta}</span>
      )}
    </WorkbenchAction>
  );

  const wrappedRow =
    actions.length > 0 ? (
      <ContextMenu>
        <ContextMenuTrigger asChild>{rowButton}</ContextMenuTrigger>
        <ContextMenuContent>
          {actions.map((action) => {
            const ActionIcon = action.icon;
            return (
              <Fragment key={action.id}>
                {action.separatorBefore && <ContextMenuSeparator />}
                <ContextMenuItem
                  disabled={action.disabled}
                  title={action.title}
                  className={
                    action.destructive ? "text-destructive focus:text-destructive" : undefined
                  }
                  onSelect={() => {
                    if (!action.disabled) {
                      action.onSelect();
                    }
                  }}
                >
                  {ActionIcon && <ActionIcon className="mr-2 h-3.5 w-3.5" />}
                  <span className="truncate">{action.label}</span>
                </ContextMenuItem>
              </Fragment>
            );
          })}
        </ContextMenuContent>
      </ContextMenu>
    ) : (
      rowButton
    );

  return (
    <div>
      <div className="flex items-center gap-1" style={{ paddingLeft: `${depth * INDENT}px` }}>
        {hasChildren ? (
          <WorkbenchIconAction
            label={isExpanded ? "Collapse" : "Expand"}
            className="size-6 flex-none text-muted-foreground"
            onClick={(event) => {
              event.stopPropagation();
              onToggle(node.id);
            }}
          >
            <ChevronRight
              className={`h-3.5 w-3.5 transition-transform ${isExpanded ? "rotate-90" : ""}`}
            />
          </WorkbenchIconAction>
        ) : reserveChevron ? (
          <span className="h-6 w-6 flex-none" />
        ) : null}
        {wrappedRow}
      </div>
      {node.children !== undefined && isExpanded && (
        <div>
          {node.children.length === 0 && node.emptyChildLabel ? (
            <p
              className="text-label text-muted-foreground"
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
                reserveChevron={childrenReserveChevron}
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
  onExpand,
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
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        onExpand?.(id);
      }
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

  const reserveChevron = nodes.some((n) => n.children !== undefined);

  return (
    <div className="space-y-1">
      {nodes.map((node) => (
        <TreeRow
          key={node.id}
          node={node}
          depth={0}
          reserveChevron={reserveChevron}
          activeId={activeId}
          expanded={expanded}
          onToggle={toggle}
        />
      ))}
    </div>
  );
};
