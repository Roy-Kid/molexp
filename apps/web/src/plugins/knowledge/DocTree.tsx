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
import type { KnowledgeSearchRow } from "@/api/generated/models/KnowledgeSearchRow";
import { StatusBadge } from "@/app/components/entity";
import { selectionForSearchHit } from "@/plugins/knowledge/searchHitSelection";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import { workspaceApi } from "@/app/state/api";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import { Code as InlineCode } from "@/components/ui/code";
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
import {
  WorkbenchAction,
  WorkbenchDismissAction,
  WorkbenchIconAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
} from "@/components/workbench";
import { cn } from "@/lib/utils";
import { buildDocTree, type DocEntityKind, type DocTreeNode } from "./knowledgeDocTree";
import { useKnowledgeDocs, useKnowledgeFacets } from "./useKnowledgeDocs";

interface KnowledgeFilterProps {
  tags: string[];
  statuses: string[];
  tag: string | null;
  status: string | null;
  disabled?: boolean;
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
  disabled = false,
  onTagChange,
  onStatusChange,
}: KnowledgeFilterProps): JSX.Element => {
  const active = tag !== null || status !== null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <WorkbenchIconAction label="Filter knowledge" disabled={disabled}>
          <Filter className="h-3.5 w-3.5" />
          {active && <span className="h-1.5 w-1.5 rounded-full bg-info" />}
        </WorkbenchIconAction>
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
  const { notes, loading, error, reload, createDoc, renameDoc, moveDoc, deleteDoc } =
    useKnowledgeDocs({ tag, status });
  const {
    tags,
    statuses,
    loading: facetsLoading,
    error: facetsError,
    reload: reloadFacets,
  } = useKnowledgeFacets();
  const filtering = tag !== null || status !== null;
  // Body-aware search (vision-loop-08): a non-empty query switches the tree to
  // a flat hit list served by GET /knowledge/search (the ONE Bundle.search verb).
  const [search, setSearch] = useState("");
  const [searchHits, setSearchHits] = useState<KnowledgeSearchRow[]>([]);
  const [searchTruncated, setSearchTruncated] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchRequestVersion, setSearchRequestVersion] = useState(0);
  useEffect(() => {
    void searchRequestVersion;
    const query = search.trim();
    if (!query) {
      setSearchHits([]);
      setSearchTruncated(false);
      setSearchLoading(false);
      setSearchError(null);
      return;
    }
    let cancelled = false;
    setSearchLoading(true);
    setSearchError(null);
    const handle = window.setTimeout(() => {
      void workspaceApi
        .searchKnowledge(query)
        .then((response) => {
          if (cancelled) return;
          setSearchHits(response.hits);
          setSearchTruncated(response.truncated);
        })
        .catch((err) => {
          if (!cancelled) {
            setSearchError(
              err instanceof Error ? err.message : "Failed to search knowledge documents",
            );
          }
        })
        .finally(() => {
          if (!cancelled) setSearchLoading(false);
        });
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [search, searchRequestVersion]);
  const searching = search.trim().length > 0;
  const handleSearchChange = (value: string): void => {
    setSearch(value);
    setSearchError(null);
    if (value.trim()) {
      setSearchLoading(true);
      return;
    }
    setSearchLoading(false);
    setSearchHits([]);
    setSearchTruncated(false);
  };
  const { prompt, dialog: promptDialog } = usePrompt();
  const { confirm, dialog: confirmDialog } = useConfirm();
  const [operationLabel, setOperationLabel] = useState<string | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const [operationSuccess, setOperationSuccess] = useState<string | null>(null);
  const [operationRetry, setOperationRetry] = useState<{
    runningLabel: string;
    successLabel: string;
    run: () => Promise<void>;
  } | null>(null);

  useEffect(() => {
    if (!operationSuccess) return;
    const handle = window.setTimeout(() => setOperationSuccess(null), 3000);
    return () => window.clearTimeout(handle);
  }, [operationSuccess]);

  const tree = buildDocTree(notes.map((n) => ({ relPath: n.relPath, name: n.name })));

  const guard = async (
    runningLabel: string,
    successLabel: string,
    run: () => Promise<void>,
  ): Promise<void> => {
    setOperationLabel(runningLabel);
    setOperationError(null);
    setOperationSuccess(null);
    setOperationRetry(null);
    try {
      await run();
      setOperationSuccess(successLabel);
    } catch (err) {
      setOperationError(err instanceof Error ? err.message : String(err));
      setOperationRetry({ runningLabel, successLabel, run });
    } finally {
      setOperationLabel(null);
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
    await guard("Creating document…", "Document created.", () => createDoc(name));
  };

  const handleCreateChild = async (parentPath: string): Promise<void> => {
    const name = await prompt({
      title: "New child document",
      label: "Document name",
      description: parentPath,
      confirmLabel: "Create",
    });
    if (!name) return;
    await guard("Creating child document…", "Child document created.", () =>
      createDoc(name, parentPath),
    );
  };

  const handleRename = async (path: string, current: string): Promise<void> => {
    const name = await prompt({
      title: "Rename document",
      label: "New name",
      defaultValue: current,
      confirmLabel: "Rename",
    });
    if (!name || name === current) return;
    await guard("Renaming document…", "Document renamed.", () => renameDoc(path, name));
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
    await guard("Moving document…", "Document moved.", () => moveDoc(path, parentPath));
  };

  const handleDelete = async (path: string, name: string): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete document?",
      description: (
        <>
          Document{" "}
          <InlineCode className="rounded-control bg-muted px-1 py-1 text-label">{name}</InlineCode>{" "}
          and its child documents will be permanently removed.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    await guard("Deleting document…", "Document deleted.", () => deleteDoc(path));
  };

  const docActions = (node: DocTreeNode): TreeNodeAction[] => {
    if (node.relPath === null) return [];
    const path = node.relPath;
    return [
      {
        id: "new-child",
        label: "New child document",
        icon: FilePlus,
        disabled: operationLabel !== null,
        onSelect: () => void handleCreateChild(path),
      },
      {
        id: "rename",
        label: "Rename",
        icon: Pencil,
        disabled: operationLabel !== null,
        onSelect: () => void handleRename(path, node.name),
      },
      {
        id: "move",
        label: "Move",
        icon: FolderInput,
        disabled: operationLabel !== null,
        onSelect: () => void handleMove(path),
      },
      {
        id: "delete",
        label: "Delete",
        icon: Trash2,
        disabled: operationLabel !== null,
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
        iconClassName: "text-muted-foreground",
        labelClassName: "font-semibold",
        actions: isKb
          ? [
              {
                id: "new-doc",
                label: "New document",
                icon: Plus,
                disabled: operationLabel !== null,
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
  const emptyTitle = filtering ? "No matching documents" : "No documents yet";
  const emptyDetail = filtering
    ? "No documents match the current tag/status filter."
    : "Create a document to start your knowledge base.";
  const treeContent =
    nodes.length === 0 ? (
      <WorkbenchOperationState
        kind="empty"
        density="compact"
        title={emptyTitle}
        detail={emptyDetail}
      />
    ) : (
      <TreeView nodes={nodes} activeId={activeId} expandPath={expandPath} />
    );

  return (
    <div className="space-y-2" aria-busy={loading || searchLoading || operationLabel !== null}>
      {/* Same action density as LeftExplorer title actions (gap-0.5, compact icons). */}
      <div className="flex items-center justify-between gap-0.5">
        <KnowledgeFilter
          tags={tags}
          statuses={statuses}
          tag={tag}
          status={status}
          disabled={operationLabel !== null}
          onTagChange={setTag}
          onStatusChange={setStatus}
        />
        <WorkbenchIconAction
          label="New document"
          kind="ghost"
          disabled={operationLabel !== null}
          onClick={() => void handleCreateRoot()}
        >
          <Plus className="h-4 w-4" />
        </WorkbenchIconAction>
      </div>
      <Input
        value={search}
        onChange={(e) => handleSearchChange(e.target.value)}
        placeholder="Search notes (title, tags, body)…"
        className="h-control-compact text-label"
        aria-label="Search knowledge"
      />
      {facetsLoading && tags.length === 0 && statuses.length === 0 && (
        <WorkbenchOperationState
          kind="loading"
          density="inline"
          title="Loading knowledge filters…"
          className="px-2"
        />
      )}
      {facetsError && (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Knowledge filters unavailable"
          detail={facetsError}
          action={<WorkbenchRetryAction onClick={() => void reloadFacets()} />}
        />
      )}
      {operationLabel && (
        <WorkbenchOperationState kind="running" density="compact" title={operationLabel} />
      )}
      {operationError && (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Document operation failed"
          detail={operationError}
          action={
            <div className="flex items-center gap-2">
              {operationRetry && (
                <WorkbenchRetryAction
                  onClick={() =>
                    void guard(
                      operationRetry.runningLabel,
                      operationRetry.successLabel,
                      operationRetry.run,
                    )
                  }
                />
              )}
              <WorkbenchDismissAction
                onClick={() => {
                  setOperationError(null);
                  setOperationRetry(null);
                }}
              />
            </div>
          }
        />
      )}
      {operationSuccess && (
        <WorkbenchOperationState kind="success" density="compact" title={operationSuccess} />
      )}
      {searching ? (
        <div className="space-y-1 px-1">
          {searchLoading ? (
            <WorkbenchOperationState
              kind="loading"
              density="compact"
              title="Searching knowledge…"
              skeletonRows={3}
            />
          ) : searchError ? (
            <WorkbenchOperationState
              kind="error"
              density="compact"
              title="Knowledge search failed"
              detail={searchError}
              action={
                <WorkbenchRetryAction
                  onClick={() => setSearchRequestVersion((version) => version + 1)}
                />
              }
            />
          ) : searchHits.length === 0 ? (
            <WorkbenchOperationState
              kind="empty"
              density="compact"
              title="No matching documents"
              detail="Search covers titles, tags, paths, and note bodies."
            />
          ) : (
            <>
              <WorkbenchOperationState
                kind="success"
                density="inline"
                title={`${searchHits.length} matching document${searchHits.length === 1 ? "" : "s"}`}
                className="sr-only"
              />
              {searchHits.map((hit) => (
                <WorkbenchAction
                  kind="ghost"
                  size="content"
                  type="button"
                  key={hit.path}
                  onClick={() => onSelect(selectionForSearchHit(hit))}
                  className={cn(
                    "block w-full rounded-control px-2 py-2 text-left hover:bg-muted",
                    activeId === hit.path && "bg-muted",
                  )}
                >
                  <span className="block truncate text-body-lg text-foreground">{hit.title}</span>
                  <span className="block truncate text-micro text-muted-foreground">
                    {hit.path}
                  </span>
                  {hit.snippet && (
                    <span className="block truncate text-micro italic text-muted-foreground">
                      {hit.snippet}
                    </span>
                  )}
                </WorkbenchAction>
              ))}
            </>
          )}
          {searchTruncated && (
            <p className="px-2 text-micro text-muted-foreground">…truncated — refine the query.</p>
          )}
        </div>
      ) : loading && notes.length === 0 ? (
        <WorkbenchOperationState
          kind="loading"
          density="compact"
          title="Loading knowledge documents…"
          skeletonRows={5}
        />
      ) : error && notes.length === 0 ? (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Could not load knowledge documents"
          detail={error}
          action={<WorkbenchRetryAction onClick={() => void reload()} />}
        />
      ) : (
        <>
          {error && (
            <WorkbenchOperationState
              kind="error"
              density="compact"
              title="Could not refresh knowledge documents"
              detail={error}
              action={<WorkbenchRetryAction onClick={() => void reload()} />}
            />
          )}
          {loading ? (
            <WorkbenchOperationState
              kind="running"
              density="compact"
              title="Refreshing knowledge documents…"
            >
              {treeContent}
            </WorkbenchOperationState>
          ) : (
            treeContent
          )}
        </>
      )}
      {promptDialog}
      {confirmDialog}
    </div>
  );
};
