/**
 * VS Code "Add Folder to Workspace" quick-input.
 * Accepts a local absolute path, remote `Host:/abs`, or `@registry-name`.
 */
import { Loader2 } from "lucide-react";
import { type JSX, useCallback, useEffect, useId, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface AddWorkspaceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAdded: () => void;
}

/** Heuristic: remote if Host:/path, user@host:/path, or @registry-name. */
const detectKind = (raw: string): "local" | "remote" => {
  const s = raw.trim();
  if (s.startsWith("@")) return "remote";
  // SCP-style host:path (not Windows drive like C:\ or C:/ alone without host)
  if (/^[A-Za-z0-9._-]+@[^:]+:\//.test(s)) return "remote";
  if (/^[A-Za-z0-9._-]+:\//.test(s) && !/^[A-Za-z]:[/\\]/.test(s)) return "remote";
  return "local";
};

const shortFailure = (raw: string): string => {
  const t = raw.toLowerCase();
  if (t.includes("not found") || t.includes("404")) return "Path not found";
  if (t.includes("not a directory")) return "Not a directory";
  if (t.includes("already") || t.includes("exists")) return "Already in workspace";
  if (t.includes("target") && t.includes("not found")) return "Unknown remote target";
  return raw.length > 80 ? "Couldn't add folder" : raw || "Couldn't add folder";
};

export const AddWorkspaceDialog = ({
  open,
  onOpenChange,
  onAdded,
}: AddWorkspaceDialogProps): JSX.Element => {
  const inputId = useId();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const inputRef = useCallback(
    (node: HTMLInputElement | null) => {
      if (node && open) node.focus();
    },
    [open],
  );

  useEffect(() => {
    if (open) {
      setValue("");
      setHint(null);
      setBusy(false);
    }
  }, [open]);

  const submit = useCallback(async () => {
    const raw = value.trim();
    if (!raw || busy) return;
    setBusy(true);
    setHint(null);
    try {
      const kind = detectKind(raw);
      if (kind === "remote") {
        if (raw.startsWith("@")) {
          await workspaceApi.addServedWorkspace({
            kind: "remote",
            name: raw.slice(1),
            activate: true,
          });
        } else {
          await workspaceApi.addServedWorkspace({
            kind: "remote",
            path: raw,
            activate: true,
          });
        }
      } else {
        try {
          await workspaceApi.addServedWorkspace({
            kind: "local",
            path: raw,
            createIfMissing: false,
            activate: true,
          });
        } catch (first) {
          const msg = first instanceof Error ? first.message : String(first);
          if (msg.toLowerCase().includes("not found")) {
            // Offer create on second submit only when path missing — mirror Open Workspace.
            await workspaceApi.addServedWorkspace({
              kind: "local",
              path: raw,
              createIfMissing: true,
              activate: true,
            });
          } else {
            throw first;
          }
        }
      }
      onOpenChange(false);
      onAdded();
    } catch (err) {
      setHint(shortFailure(err instanceof Error ? err.message : String(err)));
    } finally {
      setBusy(false);
    }
  }, [value, busy, onOpenChange, onAdded]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        className={cn(
          "top-command-offset max-w-xl -translate-y-0 gap-0 overflow-hidden p-0",
          "rounded-md shadow-lg sm:max-w-xl",
        )}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <DialogTitle className="sr-only">Add Folder to Workspace</DialogTitle>
        <DialogDescription className="sr-only">
          Enter a local path, Host:/path, or @registry-name
        </DialogDescription>

        <div className="border-b border-border/80 bg-muted/40 px-3 py-1.5">
          <p className="truncate font-mono text-micro text-muted-foreground">
            Add Folder to Workspace
            <span className="text-foreground/50"> · path or Host:/path</span>
          </p>
        </div>

        <form
          className="relative flex items-center"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label htmlFor={inputId} className="sr-only">
            Workspace path
          </label>
          <Input
            id={inputId}
            ref={inputRef}
            value={value}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
            placeholder="/path/to/workspace  or  Host:/abs/path"
            className="h-11 rounded-none border-0 bg-transparent px-3 font-mono text-body shadow-none focus-visible:ring-0"
            onChange={(e) => {
              setValue(e.target.value);
              if (hint) setHint(null);
            }}
          />
          {busy ? (
            <Loader2 className="mol-motion-progress-spin mr-3 h-4 w-4 flex-none text-muted-foreground" />
          ) : null}
        </form>

        {hint ? (
          <p className="border-t border-border/60 px-3 py-1.5 text-micro text-destructive">
            {hint}
          </p>
        ) : (
          <p className="border-t border-border/40 px-3 py-1.5 text-micro text-muted-foreground">
            Enter ↵ · Esc cancel · remote: Host:/path or @name
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
};
