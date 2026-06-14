import { Pause, Play, Terminal } from "lucide-react";
import type { JSX } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { molqApi } from "@/plugins/molq/api";

const MAX_LINES = 2_000;

interface MolqLogPanelProps {
  target: string;
  jobId: string;
}

export const MolqLogPanel = ({ target, jobId }: MolqLogPanelProps): JSX.Element => {
  const [lines, setLines] = useState<string[]>([]);
  const [paused, setPaused] = useState<boolean>(false);
  const [closed, setClosed] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pausedRef = useRef<boolean>(false);
  pausedRef.current = paused;

  useEffect(() => {
    setLines([]);
    setClosed(false);
    setError(null);

    const source = molqApi.streamLogs(target, jobId);

    source.addEventListener("message", (event) => {
      if (pausedRef.current) {
        return;
      }
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { line?: string };
        if (typeof payload.line !== "string") {
          return;
        }
        if (payload.line === "[stream closed]") {
          setClosed(true);
          source.close();
          return;
        }
        setLines((prev) => {
          const next = prev.concat(payload.line as string);
          // Keep memory bounded — drop the oldest lines once over the cap.
          return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Bad log payload");
      }
    });

    source.addEventListener("error", () => {
      setError("Stream disconnected");
      source.close();
    });

    return () => {
      source.close();
    };
  }, [target, jobId]);

  // Auto-scroll on new lines unless the user has scrolled away.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `lines` is an intentional trigger — the effect must re-run on every appended line even though the body only reads the container ref.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (nearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines]);

  const togglePause = useCallback(() => {
    setPaused((prev) => !prev);
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-zinc-950 text-zinc-100">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-1 font-mono text-[11px] text-zinc-400">
        <div className="flex items-center gap-1.5">
          <Terminal className="h-3 w-3" />
          <span>stdout</span>
          {closed && <span className="text-zinc-500">· closed</span>}
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
          onClick={togglePause}
          title={paused ? "Resume" : "Pause"}
          disabled={closed}
        >
          {paused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
        </Button>
      </div>
      <div ref={containerRef} className="flex-1 overflow-auto px-3 py-2 font-mono text-[11px]">
        {error && <div className="text-rose-300">{error}</div>}
        {lines.length === 0 && !error && (
          <div className="italic text-zinc-500">Waiting for output…</div>
        )}
        {lines.map((line, idx) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: log rows are stateless text in an append-only (cap-truncated) stream — index identity cannot corrupt reconciliation state.
          <div key={idx} className="whitespace-pre-wrap break-words">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};
