import { Tag } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import { StatusBadge } from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { Badge } from "@/components/ui/badge";

/**
 * Document-level tag + status controls for a Note.
 *
 * Reads the note's 05 metadata (`tags` / `status`) through
 * `workspaceApi.listKnowledge` — the generated client — and finds this note by
 * its bundle-relative path. This surface is **read-only**: 06 shipped no
 * tag/status *write* route (the workspace layer has `Note.set_tags` /
 * `set_status`, but no server endpoint was wired), so there is nothing to
 * persist to yet. Adding/removing tags happens through the workspace / CLI; the
 * knowledge tree filter (LeftPanel) is the interactive counterpart that acts on
 * these same fields.
 */
export const DocumentControls = ({ relPath }: { relPath: string }): JSX.Element | null => {
  const [summary, setSummary] = useState<NoteSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    workspaceApi
      .listKnowledge()
      .then((response) => {
        if (!cancelled) {
          setSummary(response.notes.find((note) => note.relPath === relPath) ?? null);
        }
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Failed to load doc metadata.");
      });
    return () => {
      cancelled = true;
    };
  }, [relPath]);

  if (error) {
    return <p className="text-xs text-destructive">{error}</p>;
  }

  const tags = summary?.tags ?? [];
  const status = summary?.status ?? null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {status ? (
        <StatusBadge status={status} size="sm" />
      ) : (
        <span className="text-[11px] text-muted-foreground">No status</span>
      )}
      <span className="text-border">·</span>
      <Tag className="h-3.5 w-3.5 text-muted-foreground" />
      {tags.length > 0 ? (
        tags.map((tag) => (
          <Badge key={tag} variant="outline" className="px-1.5 py-0 text-[10px] font-medium">
            {tag}
          </Badge>
        ))
      ) : (
        <span className="text-[11px] text-muted-foreground">No tags</span>
      )}
      <span
        className="ml-1 text-[10px] italic text-muted-foreground/70"
        title="Tags & status are set through the workspace / CLI; no write endpoint is exposed yet."
      >
        read-only
      </span>
    </div>
  );
};
