/**
 * Chat answer rendering: molplot fences + markdown body.
 */

import { type JSX, useMemo } from "react";
import { AgentPlotChart } from "@/app/renderers/agent/AgentPlotChart";
import { MarkdownContent } from "@/components/ui/markdown";
import { splitChatAnswer } from "@/lib/chat-answer-parse";

export type { ChatAnswerSegment } from "@/lib/chat-answer-parse";
export { splitChatAnswer } from "@/lib/chat-answer-parse";

export const ChatAnswerBody = ({
  text,
  linkify,
}: {
  text: string;
  linkify?: (s: string) => string;
}): JSX.Element => {
  const segments = useMemo(() => splitChatAnswer(text), [text]);
  return (
    <div className="space-y-3">
      {segments.map((seg) => {
        if (seg.kind === "molplot") {
          return (
            <AgentPlotChart
              key={`plot-${seg.title}:${JSON.stringify(seg.spec).slice(0, 120)}`}
              title={seg.title || undefined}
              spec={seg.spec}
            />
          );
        }
        const md = linkify ? linkify(seg.text) : seg.text;
        if (!md.trim()) return null;
        return <MarkdownContent key={`md-${seg.text.slice(0, 120)}`} text={md} />;
      })}
    </div>
  );
};
