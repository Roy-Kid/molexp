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
import { type ComponentType, type JSX, useEffect, useState } from "react";
import { StatusBadge } from "@/app/components/entity";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import type { KnowledgeSearchRow } from "@/api/generated/models/KnowledgeSearchRow";
import { workspaceApi } from "@/app/state/api";
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
import { Input } from "@/components/ui/input";
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
  // Body-aware search (vision-loop-08): a non-empty query switches the tree to
  // a flat hit list served by GET /knowledge/search (the ONE Bundle.search verb).
  const [search, setSearch] = useState("");
  const [searchHits, setSearchHits] = useState<KnowledgeSearchRow[]>([]);
  const [searchTruncated, setSearchTruncated] = useState(false);
  useEffect(() => {
    const query = search.trim();
    if (!query) {
      setSearchHits([]);
      setSearchTruncated(false);
      return;
    }
    let cancelled = false;
    const handle = window.setTimeout(() => {
      void workspaceApi
        .searchKnowledge(query)
        .then((response) => {
          if (cancelled) return;
          setSearchHits(response.hits);
          setSearchTruncated(response.truncated);
        })
        .catch(() => {
          if (!cancelled) setSearchHits([]);
        });
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [search]);
  const searching = search.trim().length > 0;
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
      <div className="px-1">
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search notes (title, tags, body)…"
          className="h-7 text-xs"
          aria-label="Search knowledge"
        />
      </div>
      {error && <p className="px-2 text-xs text-destructive">{error}</p>}
      {searching ? (
        <div className="space-y-0.5 px-1">
          {searchHits.length === 0 ? (
            <p className="px-2 py-4 text-center text-xs text-muted-foreground">
              No matches — the search covers titles, tags, paths, and note bodies.
            </p>
          ) : (
            searchHits.map((hit) => (
              <button
                type="button"
                key={hit.path}
                onClick={() => onSelect({ objectType: "knowledge", objectId: hit.path })}
                className={cn(
                  "block w-full rounded-sm px-2 py-1.5 text-left hover:bg-muted",
                  activeId === hit.path && "bg-muted",
                )}
              >
                <span className="block truncate text-sm text-foreground">{hit.title}</span>
                <span className="block truncate text-[11px] text-muted-foreground">
                  {hit.path}
                </span>
                {hit.snippet && (
                  <span className="block truncate text-[11px] italic text-muted-foreground">
                    {hit.snippet}
                  </span>
                )}
              </button>
            ))
          )}
          {searchTruncated && (
            <p className="px-2 text-[11px] text-muted-foreground">…truncated — refine the query.</p>
          )}
        </div>
      ) : (
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
      )}
      {promptDialog}
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
