/**
 * VS Code quick-input style: top-centered, one line of chrome, enter to submit.
 */
import { Loader2 } from "lucide-react";
import { type JSX, useCallback, useEffect, useId, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { ServedWorkspaceSummary } from "@/app/types";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { shortWorkspaceLabel } from "@/lib/workspace-path";

export interface RemoteConnectDialogProps {
  workspace: ServedWorkspaceSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConnected: () => void;
}

const shortFailure = (raw: string): string => {
  const t = raw.toLowerCase();
  if (t === "auth" || t.includes("auth")) return "Incorrect code";
  if (t === "timeout" || t.includes("timeout")) return "Timed out";
  if (t === "unreachable" || t.includes("unreachable")) return "Host unavailable";
  if (t.includes("empty")) return "Enter a code";
  return "Couldn't connect";
};

export const RemoteConnectDialog = ({
  workspace,
  open,
  onOpenChange,
  onConnected,
}: RemoteConnectDialogProps): JSX.Element => {
  const inputId = useId();
  const [code, setCode] = useState("");
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
      setCode("");
      setHint(null);
      setBusy(false);
    }
    // Clear the OTP field when the target workspace changes.
    void workspace?.key;
  }, [open, workspace?.key]);

  const submit = useCallback(async () => {
    if (!workspace || !code.trim() || busy) return;
    setBusy(true);
    setHint(null);
    try {
      await workspaceApi.connectRemoteWorkspace(workspace.key, code.trim());
      onOpenChange(false);
      onConnected();
    } catch (err) {
      setHint(shortFailure(err instanceof Error ? err.message : String(err)));
    } finally {
      setBusy(false);
    }
  }, [workspace, code, busy, onOpenChange, onConnected]);

  const host = workspace ? shortWorkspaceLabel(workspace.label) : "remote";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        // Match command palette: pin near top, strip padding / radius chrome.
        className={cn(
          "top-command-offset max-w-xl -translate-y-0 gap-0 overflow-hidden p-0",
          "rounded-md shadow-lg sm:max-w-xl",
        )}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <DialogTitle className="sr-only">Connect to {host}</DialogTitle>
        <DialogDescription className="sr-only">
          Enter verification code for {host}
        </DialogDescription>

        {/* VS Code quick-input: muted prompt line + single field */}
        <div className="border-b border-border/80 bg-muted/40 px-3 py-1.5">
          <p className="truncate font-mono text-micro text-muted-foreground">
            {host}
            <span className="text-foreground/50"> · code</span>
          </p>
        </div>

        <form
          className="relative flex items-center"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <Input
            ref={inputRef}
            id={inputId}
            autoComplete="one-time-code"
            inputMode="numeric"
            placeholder="Verification code"
            value={code}
            disabled={busy}
            onChange={(e) => {
              setCode(e.target.value);
              if (hint) setHint(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                onOpenChange(false);
              }
            }}
            className={cn(
              "h-10 w-full rounded-none border-0 bg-transparent px-3 font-mono tracking-widest",
              "shadow-none focus-visible:ring-0",
            )}
            aria-invalid={Boolean(hint)}
            aria-describedby={hint ? `${inputId}-hint` : undefined}
          />
          {busy ? (
            <Loader2
              className="mol-motion-progress-spin pointer-events-none absolute right-3 h-3.5 w-3.5 text-muted-foreground"
              aria-hidden
            />
          ) : null}
        </form>

        {hint ? (
          <p
            id={`${inputId}-hint`}
            className="border-t border-border/60 px-3 py-1.5 font-mono text-micro text-muted-foreground"
          >
            {hint}
          </p>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};
