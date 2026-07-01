import {
  Blocks,
  BookOpen,
  Check,
  FilePlus,
  Filter,
  FlaskConical,
  Folder,
  FolderInput,
  NotebookPen,
  Pencil,
  PlayCircle,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import { type ComponentType, type JSX, useState } from "react";
import { StatusBadge } from "@/app/components/entity";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { buildDocTree, type DocEntityKind, type DocTreeNode } from "./knowledgeDocTree";
import { useKnowledgeDocs, useKnowledgeFacets } from "./useKnowledgeDocs";

interface KnowledgeFilterProps {
  tags: string[];
  statuses: string[];
  tag: string | null;
  status: string | null;
  onTagChange: (tag: string | null) => void;
  onStatusChange: (status: string | null) => void;
}

/**
 * Tag/status filter for the knowledge tree — a popover command list of the
 * available facets. Selecting a facet drives `listKnowledge(?tag=&status=)`
 * server-side (06 query support) so the tree narrows to matching notes; the
 * same selection toggles off to clear.
 */
const KnowledgeFilter = ({
  tags,
  statuses,
  tag,
  status,
  onTagChange,
  onStatusChange,
}: KnowledgeFilterProps): JSX.Element => {
  const active = tag !== null || status !== null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs">
          <Filter className="h-3.5 w-3.5" /> Filter
          {active && <span className="h-1.5 w-1.5 rounded-full bg-info" />}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-56 p-0">
        <Command>
          <CommandInput placeholder="Filter by tag / status…" />
          <CommandList>
            <CommandEmpty>No facets yet.</CommandEmpty>
            {statuses.length > 0 && (
              <CommandGroup heading="Status">
                {statuses.map((s) => (
                  <CommandItem
                    key={s}
                    value={`status ${s}`}
                    onSelect={() => onStatusChange(status === s ? null : s)}
                  >
                    <Check className={cn("h-4 w-4", status === s ? "opacity-100" : "opacity-0")} />
                    <StatusBadge status={s} size="sm" />
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            {tags.length > 0 && (
              <CommandGroup heading="Tags">
                {tags.map((t) => (
                  <CommandItem
                    key={t}
                    value={`tag ${t}`}
                    onSelect={() => onTagChange(tag === t ? null : t)}
                  >
                    <Check className={cn("h-4 w-4", tag === t ? "opacity-100" : "opacity-0")} />
                    <span className="truncate">{t}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            {active && (
              <>
                <CommandSeparator />
                <CommandGroup>
                  <CommandItem
                    value="clear filters"
                    onSelect={() => {
                      onTagChange(null);
                      onStatusChange(null);
                    }}
                  >
                    <X className="h-4 w-4" /> Clear filters
                  </CommandItem>
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

interface DocTreeProps {
  snapshot: WorkspaceSnapshot;
  /** The currently-selected Note's bundle-relative path (drives active row). */
  activeId?: string;
  onSelect: (selection: Selection) => void;
}

const ENTITY_ICON: Record<DocEntityKind, ComponentType<{ className?: string }>> = {
  project: Blocks,
  experiment: FlaskConical,
  run: PlayCircle,
};

const collectExpandIds = (nodes: DocTreeNode[], acc: string[]): string[] => {
  for (const node of nodes) {
    if (node.children.length > 0) {
      acc.push(node.id);
      collectExpandIds(node.children, acc);
    }
  }
  return acc;
};

export const DocTree = ({ snapshot, activeId, onSelect }: DocTreeProps): JSX.Element => {
  const [tag, setTag] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const { notes, error, createDoc, renameDoc, moveDoc, deleteDoc } = useKnowledgeDocs({
    tag,
    status,
  });
  const { tags, statuses } = useKnowledgeFacets();
  const filtering = tag !== null || status !== null;
  const { prompt, dialog: promptDialog } = usePrompt();
  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();

  const tree = buildDocTree(notes.map((n) => ({ relPath: n.relPath, name: n.name })));

  const guard = async (label: string, run: () => Promise<void>): Promise<void> => {
    try {
      await run();
    } catch (err) {
      await alert({
        title: label,
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const handleCreateRoot = async (): Promise<void> => {
    const name = await prompt({
      title: "New document",
      label: "Document name",
      placeholder: "My note",
      confirmLabel: "Create",
    });
    if (!name) return;
    await guard("Failed to create document", () => createDoc(name));
  };

  const handleCreateChild = async (parentPath: string): Promise<void> => {
    const name = await prompt({
      title: "New child document",
      label: "Document name",
      description: parentPath,
      confirmLabel: "Create",
    });
    if (!name) return;
    await guard("Failed to create document", () => createDoc(name, parentPath));
  };

  const handleRename = async (path: string, current: string): Promise<void> => {
    const name = await prompt({
      title: "Rename document",
      label: "New name",
      defaultValue: current,
      confirmLabel: "Rename",
    });
    if (!name || name === current) return;
    await guard("Failed to rename document", () => renameDoc(path, name));
  };

  const handleMove = async (path: string): Promise<void> => {
    const parentPath = await prompt({
      title: "Move document",
      label: "New parent path",
      description: "The bundle-relative path of the parent document.",
      placeholder: "kb/parent-note",
      confirmLabel: "Move",
    });
    if (!parentPath) return;
    await guard("Failed to move document", () => moveDoc(path, parentPath));
  };

  const handleDelete = async (path: string, name: string): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete document?",
      description: (
        <>
          Document <code className="rounded bg-muted px-1 py-0.5 text-xs">{name}</code> and its
          child documents will be permanently removed.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    await guard("Failed to delete document", () => deleteDoc(path));
  };

  const docActions = (node: DocTreeNode): TreeNodeAction[] => {
    if (node.relPath === null) return [];
    const path = node.relPath;
    return [
      {
        id: "new-child",
        label: "New child document",
        icon: FilePlus,
        onSelect: () => void handleCreateChild(path),
      },
      {
        id: "rename",
        label: "Rename",
        icon: Pencil,
        onSelect: () => void handleRename(path, node.name),
      },
      {
        id: "move",
        label: "Move",
        icon: FolderInput,
        onSelect: () => void handleMove(path),
      },
      {
        id: "delete",
        label: "Delete",
        icon: Trash2,
        destructive: true,
        separatorBefore: true,
        onSelect: () => void handleDelete(path, node.name),
      },
    ];
  };

  const entityLabel = (kind: DocEntityKind, id: string): string => {
    if (kind === "project") return snapshot.projects.find((p) => p.id === id)?.name ?? id;
    if (kind === "experiment") return snapshot.experiments.find((e) => e.id === id)?.name ?? id;
    return snapshot.runs.find((r) => r.id === id)?.name ?? id;
  };

  const toTreeNode = (node: DocTreeNode): TreeNode => {
    if (node.kind === "doc" && node.relPath) {
      const path = node.relPath;
      return {
        id: node.id,
        label: node.name,
        icon: NotebookPen,
        iconClassName: "text-muted-foreground",
        onSelect: () => onSelect({ objectType: "knowledge", objectId: path }),
        actions: docActions(node),
        children: node.children.length > 0 ? node.children.map(toTreeNode) : undefined,
      };
    }
    if (node.kind === "group") {
      const isKb = node.entity === undefined;
      const label = node.entity ? entityLabel(node.entity.kind, node.entity.id) : node.name;
      const icon = node.entity ? ENTITY_ICON[node.entity.kind] : BookOpen;
      return {
        id: node.id,
        label,
        icon,
        iconClassName: isKb ? "text-blue-500" : "text-muted-foreground",
        labelClassName: "font-semibold",
        actions: isKb
          ? [
              {
                id: "new-doc",
                label: "New document",
                icon: Plus,
                onSelect: () => void handleCreateRoot(),
              },
            ]
          : undefined,
        children: node.children.map(toTreeNode),
      };
    }
    // Intermediate directory segment (no Note of its own).
    return {
      id: node.id,
      label: node.name,
      icon: Folder,
      iconClassName: "text-muted-foreground",
      children: node.children.map(toTreeNode),
    };
  };

  const nodes = tree.map(toTreeNode);
  const expandPath = collectExpandIds(tree, []);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-1 px-1">
        <KnowledgeFilter
          tags={tags}
          statuses={statuses}
          tag={tag}
          status={status}
          onTagChange={setTag}
          onStatusChange={setStatus}
        />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs"
          onClick={() => void handleCreateRoot()}
        >
          <Plus className="h-3.5 w-3.5" /> New doc
        </Button>
      </div>
      {error && <p className="px-2 text-xs text-destructive">{error}</p>}
      <TreeView
        nodes={nodes}
        activeId={activeId}
        expandPath={expandPath}
        emptyTitle={filtering ? "No matching documents" : "No documents yet"}
        emptyDescription={
          filtering
            ? "No documents match the current tag/status filter."
            : "Create a document to start your knowledge base."
        }
        emptyIcon={<BookOpen className="h-8 w-8" />}
      />
      {promptDialog}
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
