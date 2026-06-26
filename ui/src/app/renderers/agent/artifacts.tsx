import type { JSX } from "react";
import { MolplotRawChart } from "@/plugins/molplot";

// ---------------------------------------------------------------------------
// Artifact body (inline plot / table / text)
//
// Shared by the conversation transcript (artifacts folded into a tool result)
// and the Deliverables panel (a chat session's artifacts pulled out for review).
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
    // Agent-emitted specs are untyped — we accept whatever they emit
    // and let plotly validate at runtime via MolplotRawChart.
    const data = Array.isArray(inner.data) ? (inner.data as unknown[]) : [];
    const layout = (inner.layout as Record<string, unknown> | undefined) ?? {};
    return (
      <div className="space-y-2 rounded-md border border-border/60 bg-card p-3">
        {title && <p className="text-xs font-medium text-foreground">{title}</p>}
        <MolplotRawChart
          spec={{
            data,
            layout: { autosize: true, margin: { l: 48, r: 16, t: 16, b: 40 }, ...layout },
            config: { displayModeBar: false, responsive: true },
          }}
          style={{ width: "100%", height: 300 }}
        />
      </div>
    );
  }

  if (kind === "table") {
    const columns = Array.isArray(inner.columns) ? (inner.columns as string[]) : [];
    const rows = Array.isArray(inner.rows) ? (inner.rows as unknown[][]) : [];
    if (columns.length === 0 || rows.length === 0) return null;
    return (
      <div className="overflow-x-auto rounded-md border border-border/60">
        {title && (
          <p className="border-b border-border/60 bg-muted/40 px-3 py-1 text-xs font-medium">
            {title}
          </p>
        )}
        <table className="w-full text-xs">
          <thead className="bg-muted/30">
            <tr>
              {columns.map((c) => (
                <th key={`col-${c}`} className="px-3 py-1.5 text-left font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 50).map((row) => {
              const rowKey = row.map((value) => String(value ?? "")).join("|");
              return (
                <tr key={`row-${rowKey}`} className="border-t border-border/40">
                  {columns.map((column, colIdx) => (
                    <td key={`cell-${column}`} className="px-3 py-1 tabular-nums">
                      {String(row[colIdx] ?? "")}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length > 50 && (
          <p className="border-t border-border/40 bg-muted/20 px-3 py-1 text-[10px] text-muted-foreground">
            Showing 50 of {rows.length} rows
          </p>
        )}
      </div>
    );
  }

  if (kind === "text" && typeof inner.body === "string") {
    return (
      <pre className="overflow-x-auto whitespace-pre-wrap rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-[11px] text-foreground">
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
export const ToolResultArtifacts = ({
  payload,
}: {
  payload: Record<string, unknown>;
}): JSX.Element | null => {
  const result = (payload.result as Record<string, unknown> | undefined) ?? payload;
  const artifacts = Array.isArray(result.artifacts)
    ? (result.artifacts as Record<string, unknown>[])
    : [];
  if (artifacts.length === 0) return null;
  return (
    <div className="space-y-2">
      {artifacts.map((artifact) => {
        // Artifacts inside a single ToolCallCompleted are append-only —
        // identity is `kind:title`, falling back to a JSON fingerprint
        // so two identical-kind artifacts still get distinct keys.
        const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
        const fingerprint = title || JSON.stringify(artifact.payload ?? artifact);
        const key = `${String(artifact.kind ?? "?")}:${fingerprint}`;
        return <ArtifactBody key={key} payload={artifact} />;
      })}
    </div>
  );
};
