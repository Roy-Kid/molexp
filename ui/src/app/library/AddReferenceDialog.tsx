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

interface AddReferenceDialogProps {
  scope: LibraryScope;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}

const splitList = (raw: string): string[] =>
  raw
    .split(/[,;]+/)
    .map((t) => t.trim())
    .filter(Boolean);

export const AddReferenceDialog = ({
  scope,
  open,
  onOpenChange,
  onCreated,
}: AddReferenceDialogProps): JSX.Element => {
  const [key, setKey] = useState("");
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [year, setYear] = useState("");
  const [arxiv, setArxiv] = useState("");
  const [doi, setDoi] = useState("");
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = (): void => {
    setKey("");
    setTitle("");
    setAuthors("");
    setYear("");
    setArxiv("");
    setDoi("");
    setTags("");
    setNote("");
    setError(null);
  };

  const handleSave = async (): Promise<void> => {
    if (!key.trim() || !title.trim()) {
      setError("A citation key and title are required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await libraryApi.addReference(scope, {
        key: key.trim(),
        title: title.trim(),
        authors: splitList(authors),
        year: year.trim() ? Number.parseInt(year, 10) : null,
        arxiv: arxiv.trim() || null,
        doi: doi.trim() || null,
        tags: splitList(tags),
        note: note.trim(),
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
          <DialogTitle>Add reference</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="ref-key">Citation key</Label>
              <Input
                id="ref-key"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="so3krates2026"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="ref-year">Year</Label>
              <Input
                id="ref-year"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="2026"
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="ref-title">Title</Label>
            <Input
              id="ref-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="8-bit QAT of an SO(3)-equivariant transformer"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ref-authors">Authors (comma separated)</Label>
            <Input
              id="ref-authors"
              value={authors}
              onChange={(e) => setAuthors(e.target.value)}
              placeholder="Frank, Unke"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="ref-arxiv">arXiv</Label>
              <Input
                id="ref-arxiv"
                value={arxiv}
                onChange={(e) => setArxiv(e.target.value)}
                placeholder="2601.02213"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="ref-doi">DOI</Label>
              <Input
                id="ref-doi"
                value={doi}
                onChange={(e) => setDoi(e.target.value)}
                placeholder="10.1063/5.0004954"
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="ref-tags">Tags (comma separated)</Label>
            <Input
              id="ref-tags"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="quantization, mlip"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ref-note">Annotation</Label>
            <Textarea
              id="ref-note"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={3}
              placeholder="Only direct precedent for MLIP quantization."
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={() => void handleSave()} disabled={saving}>
            {saving ? "Saving…" : "Add reference"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
