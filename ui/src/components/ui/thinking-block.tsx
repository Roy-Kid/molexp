import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

/**
 * Collapsible, dimmed reasoning block (spec 03).
 *
 * Default-collapsed, prefixed `💭`, dim/italic so it reads as the agent's
 * private chain-of-thought — distinct from the answer. While the turn is still
 * streaming reasoning it shows a "Thinking…" affordance. Renders nothing when
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
        className="flex items-center gap-1 text-xs italic transition-colors hover:text-foreground"
        aria-label={expanded ? "Collapse thinking" : "Expand thinking"}
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <span>💭 {streaming ? "Thinking…" : "Thinking"}</span>
      </button>
      {expanded && (
        <pre className="mt-1 whitespace-pre-wrap rounded-md bg-muted/40 px-3 py-2 text-[11px] italic text-muted-foreground">
          {thinking}
        </pre>
      )}
    </div>
  );
};
