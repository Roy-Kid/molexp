/**
 * Explicit Yes / No for archiving chat scratch work onto a formal Run.
 *
 * Chat Mode never lands by default. After a successful ``code_run``, the agent
 * should ask — this bar makes the choice one click instead of free text.
 *
 * Copy is English-first (product default). Keep keys stable if/when i18n lands.
 */

import { HardDrive, Trash2 } from "lucide-react";
import { type JSX, useState } from "react";
import { WorkbenchAction } from "@/components/workbench";

/** User message posted when the operator accepts archive. */
export const LAND_YES_MESSAGE = "Yes — archive this work onto a formal experiment / run.";

/** User message posted when the operator keeps scratch only. */
export const LAND_NO_MESSAGE = "No — keep results in agent/.scratch only; do not land.";

/** Heuristic: assistant answer is offering archive vs scratch. */
export const looksLikeLandOffer = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  // English product strings + common Chinese agent variants (until full i18n).
  return (
    /land\s+(this|into|onto)|archive\s+(this|to|onto)|formal\s+(experiment|run)/i.test(t) ||
    /want\s+to\s+(land|archive)|should\s+I\s+(land|archive)/i.test(t) ||
    /落盘|是否.*归档|需要.*归档|正式\s*(experiment|run)/i.test(t)
  );
};

export const LandDecisionBar = ({
  intro = "Archive this work onto a formal experiment / run?",
  disabled = false,
  onDecide,
}: {
  intro?: string;
  disabled?: boolean;
  onDecide: (message: string) => void | Promise<void>;
}): JSX.Element => {
  const [busy, setBusy] = useState<"yes" | "no" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (kind: "yes" | "no"): Promise<void> => {
    if (busy || disabled) return;
    setBusy(kind);
    setError(null);
    try {
      await onDecide(kind === "yes" ? LAND_YES_MESSAGE : LAND_NO_MESSAGE);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-2 rounded-control border border-border/60 bg-muted/30 px-3 py-3">
      <p className="text-body-lg text-foreground">{intro}</p>
      <div className="flex flex-wrap items-center gap-2">
        <WorkbenchAction
          kind="primary"
          size="compact"
          disabled={Boolean(busy) || disabled}
          onClick={() => void run("yes")}
        >
          <HardDrive className="h-3.5 w-3.5" />
          {busy === "yes" ? "Sending…" : "Yes · archive"}
        </WorkbenchAction>
        <WorkbenchAction
          kind="secondary"
          size="compact"
          disabled={Boolean(busy) || disabled}
          onClick={() => void run("no")}
        >
          <Trash2 className="h-3.5 w-3.5" />
          {busy === "no" ? "Sending…" : "No · keep scratch"}
        </WorkbenchAction>
      </div>
      {error ? <p className="text-label text-destructive">{error}</p> : null}
    </div>
  );
};
