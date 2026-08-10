/**
 * Bottom status bar activity region — mirrors MolVis useStatusMessage.
 *
 * Idle is blank (never "Ready"). Info/success auto-clear; warnings and
 * errors stay until dismiss or a newer report.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { formatProgressSuffix, type StatusReportType, subscribeStatus } from "@/lib/status-report";

/** How long info/success activity stays before the left region goes blank. */
const AUTO_CLEAR_MS = 5000;

export interface StatusActivity {
  /** Empty when idle — no "Ready" placeholder. */
  text: string;
  type: StatusReportType;
  /** Optional 0–100 for long-running work. */
  progress?: number;
  /** Monotonic key so the bar can re-pulse on repeated identical messages. */
  pulse: number;
}

export function useStatusMessage(): {
  activity: StatusActivity;
  dismissActivity: () => void;
} {
  const [text, setText] = useState("");
  const [type, setType] = useState<StatusReportType>("info");
  const [progress, setProgress] = useState<number | undefined>(undefined);
  const [pulse, setPulse] = useState(0);
  const statusResetTimer = useRef<number | null>(null);

  const clearTimer = useCallback(() => {
    if (statusResetTimer.current) {
      window.clearTimeout(statusResetTimer.current);
      statusResetTimer.current = null;
    }
  }, []);

  const clearActivity = useCallback(() => {
    clearTimer();
    setText("");
    setType("info");
    setProgress(undefined);
  }, [clearTimer]);

  const applyStatus = useCallback(
    (nextText: string, nextType: StatusReportType, nextProgress?: number) => {
      const trimmed = nextText.trim();
      if (!trimmed) {
        clearActivity();
        return;
      }

      setText(trimmed);
      setType(nextType);
      setProgress(nextProgress);
      setPulse((n) => n + 1);
      clearTimer();

      // Transient tips: success / info without an active progress value auto-clear.
      // `progress === 0` means "loading started" (counting) — keep it until
      // replaced. Warnings/errors persist until dismissed or replaced.
      if (
        (nextType === "info" || nextType === "success") &&
        nextProgress === undefined
      ) {
        statusResetTimer.current = window.setTimeout(() => {
          setText("");
          setType("info");
          setProgress(undefined);
          statusResetTimer.current = null;
        }, AUTO_CLEAR_MS);
      }
    },
    [clearActivity, clearTimer],
  );

  const dismissActivity = useCallback(() => {
    clearActivity();
  }, [clearActivity]);

  useEffect(() => {
    return subscribeStatus(({ text: next, type: nextType, progress: p }) => {
      applyStatus(next, nextType, p);
    });
  }, [applyStatus]);

  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  return {
    activity: {
      text: text ? `${text}${formatProgressSuffix(progress)}` : "",
      type,
      progress,
      pulse,
    },
    dismissActivity,
  };
}
