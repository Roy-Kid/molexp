import { List } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import { BacklinksPanel } from "@/app/knowledge/BacklinksPanel";
import { buildOutline, type OutlineHeading } from "@/app/knowledge/knowledgeDocTree";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";

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
  const [isNote, setIsNote] = useState<boolean>(false);

  useEffect(() => {
    if (!relPath) {
      setOutline([]);
      setBacklinks([]);
      setIsNote(false);
      return;
    }
    let cancelled = false;
    // The body drives the outline; a non-note path (e.g. a reference) 404s here
    // and simply yields no outline/backlinks panel.
    workspaceApi
      .getNote(relPath)
      .then((note) => {
        if (cancelled) return;
        setOutline(buildOutline(note.body));
        setIsNote(true);
      })
      .catch(() => {
        if (!cancelled) setIsNote(false);
      });
    workspaceApi
      .getKnowledgeBacklinks(relPath)
      .then((response) => {
        if (!cancelled) setBacklinks(response.backlinks);
      })
      .catch(() => {
        if (!cancelled) setBacklinks([]);
      });
    return () => {
      cancelled = true;
    };
  }, [relPath]);

  if (!relPath || !isNote) return null;

  return (
    <div className="space-y-4 border-b border-border/60 p-4">
      <section className="space-y-2">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <List className="h-3.5 w-3.5" /> Outline
        </h3>
        {outline.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No headings in this document.</p>
        ) : (
          <ul className="space-y-0.5">
            {outline.map((heading) => (
              <li key={`${heading.slug}-${heading.level}`}>
                <a
                  href={`#${heading.slug}`}
                  className={`block truncate rounded-sm py-0.5 text-sm text-foreground transition-colors hover:bg-muted/40 ${INDENT_BY_LEVEL[heading.level]}`}
                  title={heading.text}
                >
                  {heading.text}
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>
      <BacklinksPanel
        backlinks={backlinks}
        onNavigate={(target) => nav.setSelection({ objectType: "knowledge", objectId: target })}
      />
    </div>
  );
};
