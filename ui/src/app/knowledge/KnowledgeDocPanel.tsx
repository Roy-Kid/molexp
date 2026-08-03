import { List } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import { ApiError } from "@/api/generated";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import { BacklinksPanel } from "@/app/knowledge/BacklinksPanel";
import { buildOutline, type OutlineHeading } from "@/app/knowledge/knowledgeDocTree";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";

const INDENT_BY_LEVEL: Record<OutlineHeading["level"], string> = {
  1: "pl-2",
  2: "pl-4",
  3: "pl-6",
};

/**
 * Knowledge right-panel (inspector slot): the selected document's H1–H3 outline
 * (from the pure {@link buildOutline}) plus its clickable backlinks. With no
 * document selected it stays idle. Backlinks navigate through the shared
 * `knowledge` selection.
 */
export const KnowledgeDocPanel = ({ selection, snapshot }: RendererProps): JSX.Element | null => {
  const nav = useNavigationState(snapshot);
  const relPath = selection.objectType === "knowledge" ? selection.objectId : "";
  const [outline, setOutline] = useState<OutlineHeading[]>([]);
  const [backlinks, setBacklinks] = useState<NoteSummary[]>([]);
  const [isNote, setIsNote] = useState<boolean | null>(null);
  const [noteLoading, setNoteLoading] = useState(false);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [settledRelPath, setSettledRelPath] = useState("");
  const [backlinksLoading, setBacklinksLoading] = useState(false);
  const [backlinksError, setBacklinksError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    if (!relPath) {
      setOutline([]);
      setBacklinks([]);
      setIsNote(null);
      setNoteLoading(false);
      setNoteError(null);
      setSettledRelPath("");
      setBacklinksLoading(false);
      setBacklinksError(null);
      return;
    }
    let cancelled = false;
    setOutline([]);
    setBacklinks([]);
    setIsNote(null);
    setNoteLoading(true);
    setNoteError(null);
    setBacklinksLoading(true);
    setBacklinksError(null);
    // The body drives the outline; a non-note path (e.g. a reference) 404s here
    // and simply yields no outline/backlinks panel.
    workspaceApi
      .getNote(relPath)
      .then((note) => {
        if (cancelled) return;
        setOutline(buildOutline(note.body));
        setIsNote(true);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setIsNote(false);
          return;
        }
        setNoteError(err instanceof Error ? err.message : "Failed to load knowledge document");
      })
      .finally(() => {
        if (!cancelled) {
          setNoteLoading(false);
          setSettledRelPath(relPath);
        }
      });
    workspaceApi
      .getKnowledgeBacklinks(relPath)
      .then((response) => {
        if (!cancelled) setBacklinks(response.backlinks);
      })
      .catch((err) => {
        if (!cancelled) {
          setBacklinksError(err instanceof Error ? err.message : "Failed to load backlinks");
        }
      })
      .finally(() => {
        if (!cancelled) setBacklinksLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [relPath, requestVersion]);

  if (!relPath) return null;

  if (noteLoading || settledRelPath !== relPath) {
    return (
      <WorkbenchOperationState
        kind="loading"
        density="compact"
        title="Loading document outline…"
        skeletonRows={3}
      />
    );
  }

  if (noteError) {
    return (
      <WorkbenchOperationState
        kind="error"
        density="compact"
        title="Could not load document inspector"
        detail={noteError}
        action={
          <WorkbenchAction
            kind="secondary"
            size="compact"
            onClick={() => {
              setNoteLoading(true);
              setRequestVersion((version) => version + 1);
            }}
          >
            Retry
          </WorkbenchAction>
        }
      />
    );
  }

  if (!isNote) return null;

  return (
    <div className="space-y-4 border-b border-border/60 p-4">
      <section className="space-y-2">
        <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <List className="h-3.5 w-3.5" /> Outline
        </h3>
        {outline.length === 0 ? (
          <WorkbenchOperationState
            kind="empty"
            density="compact"
            title="No headings"
            detail="This document has no H1–H3 headings."
          />
        ) : (
          <ul className="space-y-1">
            {outline.map((heading) => (
              <li key={`${heading.slug}-${heading.level}`}>
                <a
                  href={`#${heading.slug}`}
                  className={`block truncate rounded-sm py-1 text-sm text-foreground transition-colors hover:bg-muted/40 ${INDENT_BY_LEVEL[heading.level]}`}
                  title={heading.text}
                >
                  {heading.text}
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>
      {backlinksLoading ? (
        <WorkbenchOperationState
          kind="loading"
          density="compact"
          title="Loading backlinks…"
          skeletonRows={2}
        />
      ) : backlinksError ? (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Could not load backlinks"
          detail={backlinksError}
          action={
            <WorkbenchAction
              kind="secondary"
              size="compact"
              onClick={() => {
                setNoteLoading(true);
                setRequestVersion((version) => version + 1);
              }}
            >
              Retry
            </WorkbenchAction>
          }
        />
      ) : (
        <BacklinksPanel
          backlinks={backlinks}
          onNavigate={(target) => nav.setSelection({ objectType: "knowledge", objectId: target })}
        />
      )}
    </div>
  );
};
