/**
 * Pure parse of chat assistant text for molplot fences (no React).
 */

export type ChatAnswerSegment =
  | { kind: "markdown"; text: string }
  | { kind: "molplot"; title: string; spec: Record<string, unknown> };

/** Extract ```molplot title="…" … ``` fences; remainder stays markdown. */
export const splitChatAnswer = (text: string): ChatAnswerSegment[] => {
  const fence =
    /```molplot(?:[ \t]+title=(?:"([^"]*)"|'([^']*)'|([^\s`]+)))?[ \t]*\n([\s\S]*?)```/gi;
  const segments: ChatAnswerSegment[] = [];
  let last = 0;
  let m = fence.exec(text);
  while (m !== null) {
    if (m.index > last) {
      segments.push({ kind: "markdown", text: text.slice(last, m.index) });
    }
    const title = (m[1] ?? m[2] ?? m[3] ?? "").trim();
    const body = (m[4] ?? "").trim();
    try {
      const spec = JSON.parse(body) as Record<string, unknown>;
      segments.push({ kind: "molplot", title, spec });
    } catch {
      segments.push({
        kind: "markdown",
        text: `*(invalid molplot JSON${title ? `: ${title}` : ""})*\n\n\`\`\`\n${body.slice(0, 400)}\n\`\`\`\n`,
      });
    }
    last = m.index + m[0].length;
    m = fence.exec(text);
  }
  if (last < text.length) {
    segments.push({ kind: "markdown", text: text.slice(last) });
  }
  if (segments.length === 0) {
    segments.push({ kind: "markdown", text });
  }
  return segments;
};
