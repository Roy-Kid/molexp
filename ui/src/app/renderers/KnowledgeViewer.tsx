import { ArrowLeft, BookOpen, ExternalLink, FileText, NotebookPen } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import type { KnowledgeListResponse } from "@/api/generated/models/KnowledgeListResponse";
import type { NoteDetailResponse } from "@/api/generated/models/NoteDetailResponse";
import type { ReferenceSummary } from "@/api/generated/models/ReferenceSummary";
import { EmptyState, EntityHeader } from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";
import { MarkdownContent } from "@/components/ui/markdown";

const COLUMN = "mx-auto w-full max-w-3xl";

const formatReference = (ref: ReferenceSummary): string => {
  const authors =
    ref.authors && ref.authors.length > 0
      ? ref.authors.length > 3
        ? `${ref.authors.slice(0, 3).join(", ")} et al.`
        : ref.authors.join(", ")
      : "";
  const bits = [authors, ref.year ? `(${ref.year})` : "", ref.venue ?? ""].filter(Boolean);
  return bits.join(" · ");
};

/**
 * Knowledge browser — the workspace's OKF Concepts (Notes + literature
 * References). With no selection it lists everything; selecting a note's
 * bundle-relative path opens its narrative (``index.md``). Read-only: authoring
 * happens through the workspace / CLI, not the browser.
 */
export const KnowledgeViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const nav = useNavigationState(snapshot);
  const [data, setData] = useState<KnowledgeListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<NoteDetailResponse | null>(null);
  const [noteError, setNoteError] = useState<string | null>(null);

  const relPath = selection.objectId;

  useEffect(() => {
    let cancelled = false;
    workspaceApi
      .listKnowledge()
      .then((r) => {
        if (!cancelled) setData(r);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load knowledge.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The selected reference (if the path names one) comes from the list directly.
  const selectedReference = useMemo(
    () => data?.references.find((r) => r.relPath === relPath) ?? null,
    [data, relPath],
  );
  const isNotePath = useMemo(
    () => Boolean(relPath) && Boolean(data?.notes.some((n) => n.relPath === relPath)),
    [data, relPath],
  );

  // Fetch the note body when a note path is selected.
  useEffect(() => {
    if (!isNotePath || !relPath) {
      setNote(null);
      return;
    }
    let cancelled = false;
    setNoteError(null);
    workspaceApi
      .getNote(relPath)
      .then((n) => {
        if (!cancelled) setNote(n);
      })
      .catch((err) => {
        if (!cancelled) setNoteError(err instanceof Error ? err.message : "Failed to load note.");
      });
    return () => {
      cancelled = true;
    };
  }, [isNotePath, relPath]);

  const back = (): void => nav.setSelection({ objectType: "knowledge", objectId: "" });

  // --- Note detail --------------------------------------------------------
  if (relPath && isNotePath) {
    return (
      <div className="flex h-full flex-col bg-background">
        <EntityHeader
          icon={NotebookPen}
          title={note?.name ?? relPath}
          actions={
            <Button variant="ghost" size="sm" onClick={back}>
              <ArrowLeft className="h-4 w-4" /> Back
            </Button>
          }
        />
        <div className={`${COLUMN} flex-1 overflow-auto px-4 py-6 md:px-8`}>
          {noteError ? (
            <p className="text-sm text-destructive">{noteError}</p>
          ) : note ? (
            <MarkdownContent text={note.body || "_(empty note)_"} />
          ) : (
            <p className="text-sm italic text-muted-foreground">Loading…</p>
          )}
        </div>
      </div>
    );
  }

  // --- Reference detail ---------------------------------------------------
  if (relPath && selectedReference) {
    const ref = selectedReference;
    return (
      <div className="flex h-full flex-col bg-background">
        <EntityHeader
          icon={FileText}
          title={ref.title ?? ref.name}
          actions={
            <Button variant="ghost" size="sm" onClick={back}>
              <ArrowLeft className="h-4 w-4" /> Back
            </Button>
          }
        />
        <div className={`${COLUMN} flex-1 space-y-2 overflow-auto px-4 py-6 text-sm md:px-8`}>
          <p className="text-muted-foreground">{formatReference(ref)}</p>
          {ref.doi && (
            <p>
              DOI:{" "}
              <a
                className="text-info hover:underline"
                href={`https://doi.org/${ref.doi}`}
                target="_blank"
                rel="noreferrer"
              >
                {ref.doi}
              </a>
            </p>
          )}
          {ref.url && (
            <p>
              <a
                className="inline-flex items-center gap-1 text-info hover:underline"
                href={ref.url}
                target="_blank"
                rel="noreferrer"
              >
                <ExternalLink className="h-3.5 w-3.5" /> {ref.url}
              </a>
            </p>
          )}
          <p className="text-xs text-muted-foreground">source: {ref.source}</p>
        </div>
      </div>
    );
  }

  // --- Browse overview ----------------------------------------------------
  const notes = data?.notes ?? [];
  const references = data?.references ?? [];
  const empty = data !== null && notes.length === 0 && references.length === 0;

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        icon={BookOpen}
        title="Knowledge"
        subtitle="Notes and literature references for this workspace (OKF concepts)."
      />
      <div className={`${COLUMN} flex-1 space-y-6 overflow-auto px-4 py-6 md:px-8`}>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {empty && (
          <EmptyState
            icon={<BookOpen className="h-6 w-6" />}
            title="No knowledge yet"
            description="Notes and references mounted anywhere in the workspace appear here. Add them through the workspace or CLI."
          />
        )}

        {notes.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Notes ({notes.length})
            </h3>
            <ul className="overflow-hidden rounded-lg border border-border/60 bg-card">
              {notes.map((n) => (
                <li key={n.relPath} className="border-b border-border/50 last:border-b-0">
                  <button
                    type="button"
                    onClick={() =>
                      nav.setSelection({ objectType: "knowledge", objectId: n.relPath })
                    }
                    className="flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-muted/40"
                  >
                    <NotebookPen className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-foreground">
                        {n.name}
                      </span>
                      <span className="block truncate text-xs text-muted-foreground">
                        {n.excerpt.replace(/\n+/g, " ").trim() || "(empty note)"}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {references.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              References ({references.length})
            </h3>
            <ul className="overflow-hidden rounded-lg border border-border/60 bg-card">
              {references.map((r) => (
                <li key={r.relPath} className="border-b border-border/50 last:border-b-0">
                  <button
                    type="button"
                    onClick={() =>
                      nav.setSelection({ objectType: "knowledge", objectId: r.relPath })
                    }
                    className="flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-muted/40"
                  >
                    <FileText className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-foreground">
                        {r.title ?? r.name}
                      </span>
                      <span className="block truncate text-xs text-muted-foreground">
                        {formatReference(r)}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
};
