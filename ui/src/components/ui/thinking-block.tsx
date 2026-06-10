import { Brain, ChevronDown, ChevronRight } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";

/**
 * Collapsible, dimmed reasoning block.
 *
 * Default-collapsed so reasoning reads as the agent's private
 * chain-of-thought — distinct from the answer. While the turn is still
 * streaming it shows a pulsing "Thinking…" affordance; expanded, the
 * raw reasoning renders in a quiet bordered well. Renders nothing when
 * there is no reasoning text.
 */
export const ThinkingBlock = ({
  thinking,
  streaming = false,
}: {
  thinking: string;
  streaming?: boolean;
}): JSX.Element | null => {
  const [expanded, setExpanded] = useState(false);
  if (!thinking) return null;
  return (
    <div className="text-muted-foreground">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 rounded-sm px-1 py-0.5 text-xs transition-colors hover:text-foreground"
        aria-expanded={expanded}
        aria-label={expanded ? "Collapse reasoning" : "Expand reasoning"}
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Brain className={`h-3 w-3 ${streaming ? "motion-safe:animate-pulse" : ""}`} />
        <span className="italic">{streaming ? "Thinking…" : "Thought process"}</span>
      </button>
      {expanded && (
        <pre className="mt-1 whitespace-pre-wrap rounded-md border border-border/50 bg-muted/40 px-3 py-2 text-[11px] italic leading-relaxed text-muted-foreground">
          {thinking}
        </pre>
      )}
    </div>
  );
};
