// ─────────────────────────────────────────────────────────────────────────────
// GlobalCommandPalette
//
//   ⌘K / Ctrl+K        → Go to… (projects / experiments / runs / …)
//   ⌘⇧P / Ctrl+Shift+P → Commands (reload, reconnect, …)  — same as molvis
// ─────────────────────────────────────────────────────────────────────────────

import { Search, Terminal } from "lucide-react";
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
import { Input } from "@/components/ui/input";
import {
  WorkbenchAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
} from "@/components/workbench";

export interface PaletteCommand {
  id: string;
  label: string;
  detail?: string;
  run: () => void;
}

interface GlobalCommandPaletteProps {
  snapshot: WorkspaceSnapshot;
  /** Host actions for ⌘⇧P (Reload, Reconnect, …). */
  commands?: PaletteCommand[];
}

type PaletteMode = "goto" | "commands";

function scoreCommand(query: string, label: string, detail?: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 1;
  const hay = `${label} ${detail ?? ""}`.toLowerCase();
  if (hay === q) return 100;
  if (hay.startsWith(q)) return 80;
  if (hay.includes(q)) return 50;
  let i = 0;
  for (const ch of hay) {
    if (ch === q[i]) i += 1;
    if (i >= q.length) return 20;
  }
  return 0;
}

export const GlobalCommandPalette = ({
  snapshot,
  commands = [],
}: GlobalCommandPaletteProps): JSX.Element => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<PaletteMode>("goto");
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const modeRef = useRef(mode);
  const openRef = useRef(open);
  modeRef.current = mode;
  openRef.current = open;

  // ⌘K → go to; ⌘⇧P → commands (molvis / VS Code parity).
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const key = event.key.toLowerCase();
      const mod = event.metaKey || event.ctrlKey;
      if (!mod) return;
      if (key === "k" && !event.shiftKey) {
        event.preventDefault();
        if (openRef.current && modeRef.current === "goto") {
          setOpen(false);
        } else {
          setMode("goto");
          setOpen(true);
        }
      } else if (key === "p" && event.shiftKey) {
        event.preventDefault();
        if (openRef.current && modeRef.current === "commands") {
          setOpen(false);
        } else {
          setMode("commands");
          setOpen(true);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

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
  const gotoResults = useMemo(() => searchCatalog(catalog, query), [catalog, query]);

  const commandResults = useMemo(() => {
    const scored = commands
      .map((c) => ({ cmd: c, score: scoreCommand(query, c.label, c.detail) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score || a.cmd.label.localeCompare(b.cmd.label));
    return scored.map((x) => x.cmd);
  }, [commands, query]);

  const resultsLen = mode === "goto" ? gotoResults.length : commandResults.length;

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
    }
    // Re-run when mode flips so goto/command each start with a clean query.
    void mode;
  }, [open, mode]);

  const onQueryChange = (value: string): void => {
    setQuery(value);
    setActiveIndex(0);
  };

  const commitGoto = (index: number): void => {
    const entry = gotoResults[index];
    if (!entry) return;
    const path = entityPath(entry.ref, snapshot);
    if (!path) return;
    navigate(path);
    setOpen(false);
  };

  const commitCommand = (index: number): void => {
    const cmd = commandResults[index];
    if (!cmd) return;
    setOpen(false);
    cmd.run();
  };

  const commit = (index: number): void => {
    if (mode === "goto") commitGoto(index);
    else commitCommand(index);
  };

  const onInputKeyDown = (event: React.KeyboardEvent): void => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, Math.max(resultsLen - 1, 0)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      commit(activeIndex);
    }
  };

  const placeholder =
    mode === "commands" ? "Type a command…" : "Jump to a project, experiment, run…";

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="top-command-offset max-w-xl gap-0 overflow-hidden p-0"
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          inputRef.current?.focus();
        }}
      >
        <div className="flex items-center gap-2 border-b border-border px-3">
          {mode === "commands" ? (
            <Terminal className="h-4 w-4 flex-none text-muted-foreground" />
          ) : (
            <Search className="h-4 w-4 flex-none text-muted-foreground" />
          )}
          <Input
            ref={inputRef}
            role="combobox"
            aria-label={mode === "commands" ? "Run a command" : "Search workspace"}
            aria-autocomplete="list"
            aria-expanded="true"
            aria-controls="global-command-results"
            aria-activedescendant={
              resultsLen > 0 ? `global-command-option-${activeIndex}` : undefined
            }
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder={placeholder}
            className="h-control-comfortable w-full rounded-none border-0 bg-transparent px-0 text-body focus-visible:ring-0"
          />
          <kbd className="hidden flex-none rounded-control border border-border bg-muted px-1.5 py-0.5 font-mono text-micro text-muted-foreground sm:inline">
            {mode === "commands" ? "⌘⇧P" : "⌘K"}
          </kbd>
        </div>

        {mode === "goto" && knowledgeLoading && knowledgeDocs.length === 0 && (
          <WorkbenchOperationState
            kind="loading"
            density="inline"
            title="Loading knowledge entries…"
            className="border-b border-border px-3 py-2"
          />
        )}
        {mode === "goto" && knowledgeError && (
          <WorkbenchOperationState
            kind="error"
            density="compact"
            title="Knowledge entries unavailable"
            detail={`${knowledgeError} Projects, experiments, runs, workflows, assets, and agents remain searchable.`}
            action={<WorkbenchRetryAction onClick={() => void loadKnowledge()} />}
          />
        )}

        <div
          id="global-command-results"
          role="listbox"
          aria-label={mode === "commands" ? "Commands" : "Workspace search results"}
          className="max-h-80 overflow-y-auto p-1"
        >
          {mode === "commands" ? (
            commandResults.length === 0 ? (
              <WorkbenchOperationState
                kind="empty"
                density="compact"
                title="No commands"
                detail={query ? "No matching commands." : "No commands registered."}
              />
            ) : (
              commandResults.map((cmd, index) => {
                const isActive = index === activeIndex;
                return (
                  <WorkbenchAction
                    kind="ghost"
                    size="content"
                    type="button"
                    role="option"
                    id={`global-command-option-${index}`}
                    aria-selected={isActive}
                    tabIndex={-1}
                    key={cmd.id}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => commitCommand(index)}
                    className={`flex w-full items-center gap-3 rounded-control px-3 py-2 text-left ${
                      isActive ? "bg-muted" : ""
                    }`}
                  >
                    <Terminal className="h-4 w-4 flex-none text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate text-body-lg text-foreground">
                      {cmd.label}
                    </span>
                    {cmd.detail ? (
                      <span className="flex-none font-mono text-micro text-muted-foreground">
                        {cmd.detail}
                      </span>
                    ) : null}
                  </WorkbenchAction>
                );
              })
            )
          ) : resultsLen === 0 ? (
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
            gotoResults.map((entry, index) => {
              const meta = entityMeta(entry.ref.kind);
              const Icon = meta.icon;
              const isActive = index === activeIndex;
              return (
                <WorkbenchAction
                  kind="ghost"
                  size="content"
                  type="button"
                  role="option"
                  id={`global-command-option-${index}`}
                  aria-selected={isActive}
                  tabIndex={-1}
                  key={`${entry.ref.kind}:${entry.ref.id}`}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => commitGoto(index)}
                  className={`flex w-full items-center gap-3 rounded-control px-3 py-2 text-left ${
                    isActive ? "bg-muted" : ""
                  }`}
                >
                  <Icon className={`h-4 w-4 flex-none ${meta.iconClassName}`} />
                  <span className="min-w-0 flex-1 truncate text-body-lg text-foreground">
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
                </WorkbenchAction>
              );
            })
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
