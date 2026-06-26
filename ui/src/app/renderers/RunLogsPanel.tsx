/**
 * RunLogsPanel — the stdout/stderr terminal body shared by `RunViewer` and
 * `MolqRunViewer`. Renders the header strip (attempt label + "view latest")
 * and the scrolling log body; the caller owns the outer dark container so each
 * viewer can host it in its own layout (a plain panel vs. an EntityTabContent).
 */

import { Terminal } from "lucide-react";
import type { JSX } from "react";

interface RunLogsPanelProps {
  logs: { stdout?: string | null; stderr?: string | null } | null;
  logsError: string | null;
  selectedExecutionId: string | null;
  /** Caption shown after "stdout/stderr ·" (e.g. "latest attempt (#2)"). */
  attemptLabel: string;
  onViewLatest: () => void;
}

export const RunLogsPanel = ({
  logs,
  logsError,
  selectedExecutionId,
  attemptLabel,
  onViewLatest,
}: RunLogsPanelProps): JSX.Element => (
  <>
    <div className="flex items-center justify-between gap-2 border-b border-border bg-muted/40 px-3 py-1 font-mono text-[11px] text-muted-foreground">
      <div className="flex items-center gap-2">
        <Terminal className="h-3 w-3" />
        <span>stdout/stderr</span>
        <span className="text-muted-foreground/60">·</span>
        <span className="text-foreground">{attemptLabel}</span>
      </div>
      {selectedExecutionId && (
        <button
          type="button"
          className="text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          onClick={onViewLatest}
        >
          view latest
        </button>
      )}
    </div>
    <div className="flex-1 overflow-auto bg-muted/20 p-3 font-mono text-xs">
      {logsError ? (
        <div className="text-destructive">{logsError}</div>
      ) : logs ? (
        <div className="space-y-4">
          <section>
            <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
              stdout
            </div>
            <pre className="whitespace-pre-wrap text-foreground">
              {logs.stdout || "No stdout captured."}
            </pre>
          </section>
          <section>
            <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
              stderr
            </div>
            <pre className="whitespace-pre-wrap text-destructive/90">
              {logs.stderr || "No stderr captured."}
            </pre>
          </section>
        </div>
      ) : (
        <div className="italic text-muted-foreground">Loading logs...</div>
      )}
    </div>
  </>
);
