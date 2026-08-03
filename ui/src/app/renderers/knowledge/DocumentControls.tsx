import { Tag, X } from "lucide-react";
import { type JSX, type KeyboardEvent, useEffect, useMemo, useState } from "react";
import { StatusBadge } from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WorkbenchAction, WorkbenchTag } from "@/components/workbench";

/** Common lifecycle labels offered in the status select; `status` is an open
 * string on the backend, so the current value is always included as an option. */
const STATUS_OPTIONS = ["active", "draft", "archived"] as const;

/**
 * Writable document-level tag + status controls for a Note.
 *
 * Loads the note's 05 metadata (`tags` / `status`) through
 * `workspaceApi.listKnowledge` (the generated client), then persists edits via
 * `workspaceApi.updateNoteMeta` — the thin `PATCH /knowledge/doc/meta` route that
 * delegates to `Note.set_tags` / `Note.set_status` (the same verbs the CLI uses,
 * per the Python==UI invariant). Each save realigns local state with the
 * server-returned summary; a failed save surfaces an inline error and leaves the
 * prior state intact.
 */
export const DocumentControls = ({ relPath }: { relPath: string }): JSX.Element | null => {
  const [tags, setTags] = useState<string[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tagInput, setTagInput] = useState("");

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setLoaded(false);
    workspaceApi
      .listKnowledge()
      .then((response) => {
        if (cancelled) return;
        const summary = response.notes.find((note) => note.relPath === relPath) ?? null;
        setTags(summary?.tags ?? []);
        setStatus(summary?.status ?? null);
        setLoaded(true);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Failed to load doc metadata.");
      });
    return () => {
      cancelled = true;
    };
  }, [relPath]);

  const statusOptions = useMemo(() => {
    const opts = [...STATUS_OPTIONS] as string[];
    if (status && !opts.includes(status)) opts.push(status);
    return opts;
  }, [status]);

  const persist = async (patch: { tags?: string[]; status?: string }): Promise<void> => {
    setSaving(true);
    setError(null);
    try {
      const updated = await workspaceApi.updateNoteMeta(relPath, patch);
      setTags(updated.tags ?? []);
      setStatus(updated.status ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save doc metadata.");
    } finally {
      setSaving(false);
    }
  };

  const addTag = (raw: string): void => {
    const tag = raw.trim();
    setTagInput("");
    if (!tag || tags.includes(tag)) return;
    void persist({ tags: [...tags, tag] });
  };

  const removeTag = (tag: string): void => {
    void persist({ tags: tags.filter((t) => t !== tag) });
  };

  const onTagKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Enter") {
      event.preventDefault();
      addTag(tagInput);
    }
  };

  if (error && !loaded) {
    return <p className="text-xs text-destructive">{error}</p>;
  }
  if (!loaded) {
    return <p className="text-micro text-muted-foreground">Loading…</p>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Select
          value={status ?? undefined}
          onValueChange={(next) => {
            if (next !== status) void persist({ status: next });
          }}
          disabled={saving}
        >
          <SelectTrigger className="h-7 w-[130px] text-xs" aria-label="Document status">
            <SelectValue placeholder="Set status">
              {status ? <StatusBadge status={status} size="sm" /> : "Set status"}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {statusOptions.map((option) => (
              <SelectItem key={option} value={option} className="text-xs">
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-border">·</span>
        <Tag className="h-3.5 w-3.5 text-muted-foreground" />
        {tags.length > 0 ? (
          tags.map((tag) => (
            <WorkbenchTag
              key={tag}
              meaning="metadata"
              className="gap-1 px-2 py-0 text-micro font-medium"
            >
              {tag}
              <button
                type="button"
                aria-label={`Remove tag ${tag}`}
                className="text-muted-foreground hover:text-destructive disabled:opacity-50"
                onClick={() => removeTag(tag)}
                disabled={saving}
              >
                <X className="h-2.5 w-2.5" />
              </button>
            </WorkbenchTag>
          ))
        ) : (
          <span className="text-micro text-muted-foreground">No tags</span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Input
          value={tagInput}
          onChange={(event) => setTagInput(event.target.value)}
          onKeyDown={onTagKeyDown}
          placeholder="Add tag…"
          aria-label="Add tag"
          className="h-7 w-40 text-xs"
          disabled={saving}
        />
        <WorkbenchAction
          kind="secondary"
          size="compact"
          type="button"
          className="h-7 text-xs"
          onClick={() => addTag(tagInput)}
          disabled={saving || tagInput.trim().length === 0}
        >
          Add
        </WorkbenchAction>
      </div>
      {error ? <p className="text-micro text-destructive">{error}</p> : null}
    </div>
  );
};
