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
import { libraryApi } from "./api";
import type { LibraryScope } from "./types";

interface ImportZoteroDialogProps {
  scope: LibraryScope;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: () => void;
}

export const ImportZoteroDialog = ({
  scope,
  open,
  onOpenChange,
  onImported,
}: ImportZoteroDialogProps): JSX.Element => {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const handleImport = async (): Promise<void> => {
    if (!path.trim()) {
      setError("Enter the path to your Zotero library.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const { imported } = await libraryApi.importZotero(scope, path.trim());
      setResult(`Linked ${imported} reference(s) from Zotero.`);
      onImported();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Link a Zotero library</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            References are imported as records and PDFs are <strong>pointed at</strong> in Zotero's
            own <code className="rounded bg-muted px-1">storage/</code> — nothing is copied into the
            workspace.
          </p>
          <div className="space-y-1">
            <Label htmlFor="zotero-path">Zotero library path</Label>
            <Input
              id="zotero-path"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="~/Zotero  (or …/Zotero/zotero.sqlite)"
            />
            <p className="text-xs text-muted-foreground">
              The Zotero data directory or its <code>zotero.sqlite</code> file. Read-only — safe to
              run while Zotero is open.
            </p>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {result && <p className="text-sm text-success">{result}</p>}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            Close
          </Button>
          <Button onClick={() => void handleImport()} disabled={busy}>
            {busy ? "Linking…" : "Link library"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
