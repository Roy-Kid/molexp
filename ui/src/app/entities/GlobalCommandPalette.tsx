// ─────────────────────────────────────────────────────────────────────────────
// GlobalCommandPalette — ⌘K / Ctrl-K jump-to-anything across the whole
// workspace. Replaces the old ContextBar search, which only filtered the
// currently-visible left tree. Results span every entity kind and navigate via
// the single ``entityPath`` URL scheme, so this is the connective tissue that
// lets a user reach any node from anywhere.
// ─────────────────────────────────────────────────────────────────────────────

import { Search } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import { StatusBadge } from "@/app/components/entity";
import { buildCatalog, searchCatalog } from "@/app/entities/catalog";
import { entityMeta } from "@/app/entities/kinds";
import { entityPath } from "@/app/entities/paths";
import { workspaceApi } from "@/app/state/api";
import type { SemanticStatus, WorkspaceSnapshot } from "@/app/types";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";

interface GlobalCommandPaletteProps {
  snapshot: WorkspaceSnapshot;
}

export const GlobalCommandPalette = ({ snapshot }: GlobalCommandPaletteProps): JSX.Element => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Global hotkey: ⌘K (mac) / Ctrl-K (everywhere else).
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Knowledge docs join the jump list (vision-loop-08). Best-effort fetch-once
  // (the useKnowledgeFacets pattern): the palette still works without them.
  const [knowledgeDocs, setKnowledgeDocs] = useState<NoteSummary[]>([]);
  const [knowledgeLoading, setKnowledgeLoading] = useState(true);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);

  const loadKnowledge = useCallback(async (): Promise<void> => {
    setKnowledgeLoading(true);
    try {
      const response = await workspaceApi.listKnowledge();
      setKnowledgeDocs(response.notes);
      setKnowledgeError(null);
    } catch (err) {
      setKnowledgeError(
        err instanceof Error ? err.message : "Failed to load knowledge search entries.",
      );
    } finally {
      setKnowledgeLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadKnowledge();
  }, [loadKnowledge]);

  const catalog = useMemo(() => buildCatalog(snapshot, knowledgeDocs), [snapshot, knowledgeDocs]);
  const results = useMemo(() => searchCatalog(catalog, query), [catalog, query]);

  // Reset transient state whenever the dialog opens.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
    }
  }, [open]);

  const onQueryChange = (value: string): void => {
    setQuery(value);
    setActiveIndex(0);
  };

  const commit = (index: number): void => {
    const entry = results[index];
    if (!entry) return;
    const path = entityPath(entry.ref, snapshot);
    if (!path) return;
    navigate(path);
    setOpen(false);
  };

  const onInputKeyDown = (event: React.KeyboardEvent): void => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      commit(activeIndex);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="top-[20%] max-w-xl gap-0 overflow-hidden p-0"
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          inputRef.current?.focus();
        }}
      >
        <div className="flex items-center gap-2 border-b border-border px-3">
          <Search className="h-4 w-4 flex-none text-muted-foreground" />
          <input
            ref={inputRef}
            role="combobox"
            aria-label="Search workspace"
            aria-autocomplete="list"
            aria-expanded="true"
            aria-controls="global-command-results"
            aria-activedescendant={
              results.length > 0 ? `global-command-option-${activeIndex}` : undefined
            }
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="Jump to a project, experiment, run, workflow, asset, agent, note…"
            className="h-control-comfortable w-full bg-transparent text-body outline-none placeholder:text-muted-foreground"
          />
        </div>
        {knowledgeLoading && knowledgeDocs.length === 0 && (
          <WorkbenchOperationState
            kind="loading"
            density="inline"
            title="Loading knowledge entries…"
            className="border-b border-border px-3 py-2"
          />
        )}
        {knowledgeError && (
          <WorkbenchOperationState
            kind="error"
            density="compact"
            title="Knowledge entries unavailable"
            detail={`${knowledgeError} Projects, experiments, runs, workflows, assets, and agents remain searchable.`}
            action={
              <WorkbenchAction kind="secondary" size="compact" onClick={() => void loadKnowledge()}>
                Retry
              </WorkbenchAction>
            }
          />
        )}
        <div
          id="global-command-results"
          role="listbox"
          aria-label="Workspace search results"
          aria-busy={knowledgeLoading}
          className="max-h-80 overflow-y-auto p-1"
        >
          {results.length === 0 ? (
            knowledgeLoading ? (
              <WorkbenchOperationState
                kind="loading"
                density="compact"
                title="Searching available workspace entries…"
              />
            ) : (
              <WorkbenchOperationState
                kind="empty"
                density="compact"
                title="No matches"
                detail={
                  knowledgeError
                    ? "No matches in the available workspace entries."
                    : "Try another name, ID, or entity kind."
                }
              />
            )
          ) : (
            <>
              <WorkbenchOperationState
                kind="success"
                density="inline"
                title={`${results.length} result${results.length === 1 ? "" : "s"}`}
                className="sr-only"
              />
              {results.map((entry, index) => {
                const meta = entityMeta(entry.ref.kind);
                const Icon = meta.icon;
                const isActive = index === activeIndex;
                return (
                  <button
                    type="button"
                    role="option"
                    id={`global-command-option-${index}`}
                    aria-selected={isActive}
                    tabIndex={-1}
                    key={`${entry.ref.kind}:${entry.ref.id}`}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => commit(index)}
                    className={`flex w-full items-center gap-3 rounded-sm px-3 py-2 text-left ${
                      isActive ? "bg-muted" : ""
                    }`}
                  >
                    <Icon className={`h-4 w-4 flex-none ${meta.iconClassName}`} />
                    <span className="min-w-0 flex-1 truncate text-sm text-foreground">
                      {entry.ref.label ?? entry.ref.id}
                    </span>
                    <span className="flex-none text-micro uppercase tracking-wide text-muted-foreground">
                      {meta.label}
                    </span>
                    {entry.ref.status && (
                      <StatusBadge
                        status={entry.ref.status as SemanticStatus}
                        size="sm"
                        dot
                        showLabel={false}
                      />
                    )}
                  </button>
                );
              })}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
