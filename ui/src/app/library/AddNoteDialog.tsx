import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { libraryApi } from "./api";
import type { LibraryScope } from "./types";

interface AddNoteDialogProps {
  scope: LibraryScope;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}

const splitTags = (raw: string): string[] =>
  raw
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean);

export const AddNoteDialog = ({
  scope,
  open,
  onOpenChange,
  onCreated,
}: AddNoteDialogProps): JSX.Element => {
  const [title, setTitle] = useState("");
  const [summary, setSummary] = useState("");
  const [tags, setTags] = useState("");
  const [content, setContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = (): void => {
    setTitle("");
    setSummary("");
    setTags("");
    setContent("");
    setError(null);
  };

  const handleSave = async (): Promise<void> => {
    if (!title.trim()) {
      setError("A title is required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await libraryApi.addNote(scope, {
        title: title.trim(),
        content,
        summary: summary.trim(),
        tags: splitTags(tags),
      });
      reset();
      onOpenChange(false);
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>New note</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="note-title">Title</Label>
            <Input
              id="note-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Gold-standard decision"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="note-summary">Summary (one line)</Label>
            <Input
              id="note-summary"
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              placeholder="Why fp32 (not fp64) anchors ΔF"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="note-tags">Tags (comma or space separated)</Label>
            <Input
              id="note-tags"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="decision quantization"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="note-content">Markdown</Label>
            <Textarea
              id="note-content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={12}
              className="font-mono text-xs"
              placeholder="# Heading&#10;&#10;Body text…"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={() => void handleSave()} disabled={saving}>
            {saving ? "Saving…" : "Create note"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
