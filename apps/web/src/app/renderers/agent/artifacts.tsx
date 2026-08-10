import type { JSX } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AgentPlotChart } from "./AgentPlotChart";
import { InlineStructureViewer } from "./inlineStructure";

// ---------------------------------------------------------------------------
// Artifact body (inline plot / table / text)
//
// Shared by the conversation transcript (artifacts folded into a tool result)
// and the inspector Artifacts tab.
// ---------------------------------------------------------------------------

export const ArtifactBody = ({
  payload,
}: {
  payload: Record<string, unknown>;
}): JSX.Element | null => {
  const kind = String(payload.kind ?? "");
  const title = typeof payload.title === "string" ? payload.title : "";
  const inner = (payload.payload as Record<string, unknown> | undefined) ?? payload;

  if (kind === "plot") {
    return <AgentPlotChart title={title} spec={inner} />;
  }

  if (kind === "structure") {
    const content = typeof inner.content === "string" ? inner.content : "";
    const filename =
      typeof inner.filename === "string"
        ? inner.filename
        : `structure.${typeof inner.format === "string" ? inner.format : "xyz"}`;
    if (!content) return null;
    return <InlineStructureViewer content={content} filename={filename} title={title} />;
  }

  if (kind === "table") {
    const columns = Array.isArray(inner.columns) ? (inner.columns as string[]) : [];
    const rows = Array.isArray(inner.rows) ? (inner.rows as unknown[][]) : [];
    if (columns.length === 0 || rows.length === 0) return null;
    return (
      <div className="overflow-x-auto rounded-control border border-border/60">
        {title && (
          <p className="border-b border-border/60 bg-muted/40 px-3 py-1 text-label font-medium">
            {title}
          </p>
        )}
        <Table className="w-full text-label">
          <TableHeader className="bg-muted/30">
            <TableRow>
              {columns.map((c) => (
                <TableHead key={`col-${c}`} className="px-3 py-2 text-left font-medium">
                  {c}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.slice(0, 50).map((row) => {
              const rowKey = row.map((value) => String(value ?? "")).join("|");
              return (
                <TableRow key={`row-${rowKey}`} className="border-t border-border/40">
                  {columns.map((column, colIdx) => (
                    <TableCell key={`cell-${column}`} className="px-3 py-1 tabular-nums">
                      {String(row[colIdx] ?? "")}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
        {rows.length > 50 && (
          <p className="border-t border-border/40 bg-muted/20 px-3 py-1 text-micro text-muted-foreground">
            Showing 50 of {rows.length} rows
          </p>
        )}
      </div>
    );
  }

  if (kind === "text" && typeof inner.body === "string") {
    return (
      <pre className="overflow-x-auto whitespace-pre-wrap rounded-control border border-border/60 bg-muted/40 px-3 py-2 text-micro text-foreground">
        {inner.body}
      </pre>
    );
  }

  return null;
};

/**
 * Renders artifacts folded inside a ToolCallCompleted payload.
 *
 * Reads `result.artifacts` (canonical) or `payload.artifacts` (loose mock)
 * and dispatches each entry to ArtifactBody. Falls back silently when the
 * tool call carried no inline artifacts.
 */
/** Collect embed artifacts from a tool_call_completed event payload. */
export const artifactsFromPayload = (
  payload: Record<string, unknown>,
): Record<string, unknown>[] => {
  const result = (payload.result as Record<string, unknown> | undefined) ?? {};
  const raw = Array.isArray(payload.artifacts)
    ? payload.artifacts
    : Array.isArray(result.artifacts)
      ? result.artifacts
      : [];
  return raw.filter((a): a is Record<string, unknown> => Boolean(a) && typeof a === "object");
};

export const ToolResultArtifacts = ({
  payload,
}: {
  payload: Record<string, unknown>;
}): JSX.Element | null => {
  const artifacts = artifactsFromPayload(payload);
  if (artifacts.length === 0) return null;
  return (
    <div className="space-y-2">
      {artifacts.map((artifact) => {
        // Artifacts inside a single ToolCallCompleted are append-only —
        // identity is `kind:title`, falling back to a JSON fingerprint
        // so two identical-kind artifacts still get distinct keys.
        const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
        const fingerprint = title || JSON.stringify(artifact.payload ?? artifact).slice(0, 120);
        const key = `${String(artifact.kind ?? "?")}:${fingerprint}`;
        return <ArtifactBody key={key} payload={artifact} />;
      })}
    </div>
  );
};

/** All embed artifacts from a turn's step events (chat deliverables). */
export const TurnEmbedArtifacts = ({
  events,
}: {
  events: { type: string; payload?: Record<string, unknown> | null }[];
}): JSX.Element | null => {
  const artifacts: Record<string, unknown>[] = [];
  for (const ev of events) {
    if (ev.type !== "tool_call_completed") continue;
    artifacts.push(...artifactsFromPayload((ev.payload ?? {}) as Record<string, unknown>));
  }
  if (artifacts.length === 0) return null;
  return (
    <div className="space-y-2">
      {artifacts.map((artifact, i) => {
        const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
        const key = `${String(artifact.kind ?? "?")}:${title}:${i}`;
        return <ArtifactBody key={key} payload={artifact} />;
      })}
    </div>
  );
};
