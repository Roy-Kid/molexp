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
    <div className="flex items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-[11px] text-zinc-400">
      <div className="flex items-center gap-2">
        <Terminal className="h-3 w-3" />
        <span>stdout/stderr</span>
        <span className="text-zinc-500">·</span>
        <span className="text-zinc-300">{attemptLabel}</span>
      </div>
      {selectedExecutionId && (
        <button
          type="button"
          className="text-zinc-400 underline-offset-2 hover:text-zinc-100 hover:underline"
          onClick={onViewLatest}
        >
          view latest
        </button>
      )}
    </div>
    <div className="flex-1 overflow-auto p-3 font-mono text-xs">
      {logsError ? (
        <div className="text-rose-300">{logsError}</div>
      ) : logs ? (
        <div className="space-y-4">
          <section>
            <div className="mb-1 text-[11px] uppercase text-zinc-500">stdout</div>
            <pre className="whitespace-pre-wrap text-zinc-100">
              {logs.stdout || "No stdout captured."}
            </pre>
          </section>
          <section>
            <div className="mb-1 text-[11px] uppercase text-zinc-500">stderr</div>
            <pre className="whitespace-pre-wrap text-rose-100">
              {logs.stderr || "No stderr captured."}
            </pre>
          </section>
        </div>
      ) : (
        <div className="italic opacity-60">Loading logs...</div>
      )}
    </div>
  </>
);
