import {
  Archive,
  Download,
  FileJson,
  FileText,
  GitCommitHorizontal,
  Image as ImageIcon,
  Layers,
  Package,
  ScrollText,
  ShieldAlert,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { AssetLineageNode } from "@/api/generated/models/AssetLineageNode";
import type { AssetLineageResponse } from "@/api/generated/models/AssetLineageResponse";
import {
  EMPTY_COPY,
  EmptyState,
  EntityMetric,
  EntityPage,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, AssetKind, RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

// ── Helpers ────────────────────────────────────────────────────────────────

const formatBytes = (bytes: number | null | undefined): string => {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const KIND_META: Record<string, { label: string; icon: typeof Archive; accent: string }> = {
  data: { label: "Data", icon: Package, accent: "text-amber-500" },
  artifact: { label: "Artifact", icon: Archive, accent: "text-sky-500" },
  log: { label: "Log", icon: ScrollText, accent: "text-emerald-500" },
  checkpoint: {
    label: "Checkpoint",
    icon: GitCommitHorizontal,
    accent: "text-purple-500",
  },
  error_trace: { label: "Error Trace", icon: ShieldAlert, accent: "text-red-500" },
  execution_state: {
    label: "Execution State",
    icon: Layers,
    accent: "text-indigo-500",
  },
  output: { label: "Output", icon: FileJson, accent: "text-teal-500" },
};

const kindMeta = (kind: string) =>
  KIND_META[kind] ?? { label: kind, icon: Archive, accent: "text-muted-foreground" };

const extraValue = <T,>(asset: ApiAssetResponse, key: string): T | undefined =>
  (asset.extra as Record<string, unknown> | undefined)?.[key] as T | undefined;

const isTextual = (mime: string | undefined, path: string | undefined): boolean => {
  if (mime?.startsWith("text/")) return true;
  if (mime === "application/json") return true;
  const ext = path?.split(".").pop()?.toLowerCase() ?? "";
  return ["json", "yaml", "yml", "txt", "md", "py", "csv", "log"].includes(ext);
};

const isImage = (mime: string | undefined, path: string | undefined): boolean => {
  if (mime?.startsWith("image/")) return true;
  const ext = path?.split(".").pop()?.toLowerCase() ?? "";
  return ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext);
};

// ── Per-kind content panels ───────────────────────────────────────────────

const BinaryPreview = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [textContent, setTextContent] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mime = extraValue<string>(asset, "mime");
  const size = extraValue<number>(asset, "size") ?? null;
  const textual = isTextual(mime, asset.path);
  const image = isImage(mime, asset.path);

  useEffect(() => {
    if (!textual && !image) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    fetch(`/api/assets/${encodeURIComponent(asset.id)}/content`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load asset: ${response.statusText}`);
        }
        return response.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        if (image) {
          objectUrl = URL.createObjectURL(blob);
          setImageUrl(objectUrl);
        } else {
          return blob.text().then((text) => {
            if (!cancelled) setTextContent(text);
          });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load asset content");
        }
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset.id, textual, image]);

  const downloadUrl = `/api/assets/${encodeURIComponent(asset.id)}/content`;

  if (error) {
    return (
      <div className="p-6">
        <EmptyState title="Failed to load content." description={error} />
      </div>
    );
  }

  if (image) {
    return (
      <div className="flex h-full items-center justify-center bg-muted/20 p-6">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={asset.name}
            className="max-h-full max-w-full rounded-md border border-border/60"
          />
        ) : (
          <EmptyState title="Loading…" icon={<ImageIcon className="h-8 w-8" />} />
        )}
      </div>
    );
  }

  if (textual) {
    const ext = asset.path?.split(".").pop()?.toLowerCase() ?? "";
    let displayed = textContent ?? "";
    if (textContent && ext === "json") {
      try {
        displayed = JSON.stringify(JSON.parse(textContent), null, 2);
      } catch {
        // leave as-is
      }
    }
    return (
      <div className="h-full overflow-auto">
        <pre className="whitespace-pre-wrap break-words px-6 py-4 font-mono text-xs text-foreground">
          {textContent === null ? "Loading…" : displayed}
        </pre>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center">
      <EmptyState
        title="Binary content"
        description={`${formatBytes(size)} · ${mime ?? "unknown"}`}
        icon={<Package className="h-8 w-8" />}
        action={
          <a href={downloadUrl} download={asset.name}>
            <Button size="sm" variant="outline">
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          </a>
        }
      />
    </div>
  );
};

const LogTail = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [lines, setLines] = useState<string>("Loading…");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/assets/${encodeURIComponent(asset.id)}/tail?n=500`)
      .then((response) => {
        if (!response.ok) throw new Error(response.statusText);
        return response.text();
      })
      .then((text) => {
        if (!cancelled) setLines(text);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load log tail");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [asset.id]);

  if (error) {
    return (
      <div className="p-6">
        <EmptyState title="Failed to load log" description={error} />
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-black/90 px-4 py-3 font-mono text-xs text-emerald-100">
      <pre className="whitespace-pre-wrap break-words">{lines}</pre>
    </div>
  );
};

const JsonPreview = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/assets/${encodeURIComponent(asset.id)}/content`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
      .then((t) => {
        if (cancelled) return;
        try {
          setText(JSON.stringify(JSON.parse(t), null, 2));
        } catch {
          setText(t);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load");
      });
    return () => {
      cancelled = true;
    };
  }, [asset.id]);

  if (error) {
    return (
      <div className="p-6">
        <EmptyState title="Failed to load content" description={error} />
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto">
      <pre className="whitespace-pre-wrap break-words px-6 py-4 font-mono text-xs text-foreground">
        {text ?? "Loading…"}
      </pre>
    </div>
  );
};

const ErrorTraceView = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const exceptionType = extraValue<string>(asset, "exception_type");
  const message = extraValue<string>(asset, "message");
  const executionId = extraValue<string>(asset, "execution_id");

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-red-500/20 bg-red-500/5 px-6 py-4">
        <div className="text-xs font-semibold uppercase text-red-500">
          {exceptionType ?? "Unknown exception"}
        </div>
        <div className="mt-1 text-sm text-foreground">{message ?? "(no message)"}</div>
        {executionId && (
          <div className="mt-1 font-mono text-xs text-muted-foreground">
            execution_id = {executionId}
          </div>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        <BinaryPreview asset={asset} />
      </div>
    </div>
  );
};

const ContentPanel = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  switch (asset.kind) {
    case "log":
      return <LogTail asset={asset} />;
    case "checkpoint":
    case "execution_state":
      return <JsonPreview asset={asset} />;
    case "error_trace":
      return <ErrorTraceView asset={asset} />;
    default:
      return <BinaryPreview asset={asset} />;
  }
};

// ── Lineage column ────────────────────────────────────────────────────────

const LineageColumn = ({
  title,
  nodes,
  onSelect,
}: {
  title: string;
  nodes: AssetLineageNode[];
  onSelect: (assetId: string) => void;
}): JSX.Element => {
  if (nodes.length === 0) {
    return (
      <div className="rounded border border-border/70 bg-muted/10 p-3 text-xs text-muted-foreground">
        <div className="mb-1 font-semibold text-foreground">{title}</div>
        <div>—</div>
      </div>
    );
  }
  return (
    <div className="rounded border border-border/70 bg-muted/10 p-3">
      <div className="mb-2 text-xs font-semibold text-foreground">
        {title} <span className="text-muted-foreground">({nodes.length})</span>
      </div>
      <ul className="space-y-1.5">
        {nodes.map((node) => {
          const meta = kindMeta(node.kind);
          const Icon = meta.icon;
          return (
            <li key={node.id}>
              <button
                type="button"
                onClick={() => onSelect(node.id)}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-accent"
              >
                <Icon className={`h-3.5 w-3.5 shrink-0 ${meta.accent}`} />
                <span className="flex-1 truncate font-mono">{node.name}</span>
                <Badge variant="outline" className="text-[10px]">
                  {meta.label}
                </Badge>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

// ── Main viewer ────────────────────────────────────────────────────────────

export const AssetViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const [asset, setAsset] = useState<ApiAssetResponse | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [lineage, setLineage] = useState<AssetLineageResponse | null>(null);
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  const assetId = selection.objectId;

  useEffect(() => {
    let cancelled = false;
    workspaceApi
      .getAssets()
      .then((all) => {
        if (cancelled) return;
        const match = all.find((a) => a.id === assetId);
        if (!match) {
          setNotFound(true);
        } else {
          setAsset(match);
          setNotFound(false);
        }
      })
      .catch((err) => {
        if (typeof console !== "undefined") console.error("Failed to load asset", err);
        setNotFound(true);
      });
    return () => {
      cancelled = true;
    };
  }, [assetId]);

  useEffect(() => {
    let cancelled = false;
    setLineage(null);
    if (!assetId) return;
    workspaceApi
      .getAssetLineage(assetId)
      .then((res) => {
        if (cancelled) return;
        setLineage(res);
      })
      .catch((err) => {
        if (typeof console !== "undefined") console.error("Failed to load lineage", err);
      });
    return () => {
      cancelled = true;
    };
  }, [assetId]);

  const assetSummary = useMemo(
    () => snapshot.assets.find((a) => a.id === assetId),
    [snapshot.assets, assetId],
  );

  if (notFound) {
    return <div className="p-8 text-muted-foreground">Asset not found.</div>;
  }

  if (!asset) {
    return <div className="p-8 text-muted-foreground">Loading asset…</div>;
  }

  const meta = kindMeta(asset.kind);
  const size = extraValue<number>(asset, "size") ?? null;
  const mime = extraValue<string>(asset, "mime");
  const downloadUrl = `/api/assets/${encodeURIComponent(asset.id)}/content`;

  const producerRunId = asset.producer?.run_id as string | undefined;
  const producerTaskId = asset.producer?.task_id as string | undefined;
  const producerExecId = asset.producer?.execution_id as string | undefined;
  const producerRun = producerRunId ? snapshot.runs.find((r) => r.id === producerRunId) : null;

  const scopeLabel =
    asset.scope_kind === "workspace"
      ? "workspace"
      : `${asset.scope_kind}: ${asset.scope_ids.join(" / ")}`;

  const tagEntries = Object.entries(asset.tags ?? {});

  return (
    <EntityPage
      breadcrumbs={breadcrumbs}
      canNavigateUp={canNavigateUp}
      onNavigateUp={navigateUp}
      icon={meta.icon}
      title={asset.name}
      status={assetSummary?.status}
      subtitle={`${meta.label} · ${scopeLabel}`}
      actions={
        <a href={downloadUrl} download={asset.name}>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Download
          </Button>
        </a>
      }
      metrics={
        <>
          <EntityMetric label="Kind" value={meta.label} />
          <EntityMetric label="Scope" value={asset.scope_kind} />
          <EntityMetric label="Size" value={formatBytes(size)} />
        </>
      }
      tabs={[
        {
          value: "overview",
          label: "Overview",
          content: (
            <OverviewPage
              aside={
                <>
                  <OverviewSection title="Highlights">
                    <OverviewHighlightGrid>
                      <OverviewHighlight label="Kind" value={meta.label} />
                      <OverviewHighlight label="Size" value={formatBytes(size)} />
                      <OverviewHighlight
                        label="Scope"
                        value={asset.scope_kind}
                        detail={scopeLabel}
                      />
                      <OverviewHighlight label="Status" value={assetSummary?.status ?? "unknown"} />
                    </OverviewHighlightGrid>
                  </OverviewSection>

                  {(assetSummary?.projectId || producerRun) && (
                    <OverviewSection title="Relationships">
                      <div className="flex flex-wrap gap-1.5">
                        {assetSummary?.projectId && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => {
                              if (assetSummary.projectId) {
                                setSelection({
                                  objectType: "project",
                                  objectId: assetSummary.projectId,
                                });
                              }
                            }}
                          >
                            Project: {assetSummary.projectId}
                          </Button>
                        )}
                        {producerRun && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() =>
                              setSelection({ objectType: "run", objectId: producerRun.id })
                            }
                          >
                            Producer Run: {producerRun.name || producerRun.id}
                          </Button>
                        )}
                      </div>
                    </OverviewSection>
                  )}

                  {(size != null || mime) && (
                    <OverviewSection title="Payload">
                      <KeyValueGrid
                        items={[
                          { label: "MIME", value: mime ?? "—" },
                          { label: "Size", value: formatBytes(size) },
                        ]}
                      />
                    </OverviewSection>
                  )}

                  {tagEntries.length > 0 && (
                    <OverviewSection title="Tags">
                      <div className="flex flex-wrap gap-1.5">
                        {tagEntries.map(([key, value]) => (
                          <Badge key={key} variant="secondary" className="text-xs">
                            {key}: {value}
                          </Badge>
                        ))}
                      </div>
                    </OverviewSection>
                  )}
                </>
              }
            >
              <OverviewSection title="Identity">
                <KeyValueGrid
                  items={[
                    {
                      label: "Asset ID",
                      value: <span className="font-mono text-xs">{asset.id}</span>,
                    },
                    { label: "Name", value: asset.name },
                    { label: "Kind", value: meta.label },
                    {
                      label: "Scope",
                      value: <span className="font-mono text-xs">{scopeLabel}</span>,
                    },
                    {
                      label: "Path",
                      value: <span className="break-all font-mono text-xs">{asset.path}</span>,
                    },
                    {
                      label: "Created",
                      value: new Date(asset.created_at).toLocaleString(),
                    },
                    {
                      label: "Updated",
                      value: new Date(asset.updated_at).toLocaleString(),
                    },
                  ]}
                />
              </OverviewSection>

              {(producerRunId || producerTaskId || producerExecId) && (
                <OverviewSection title="Producer">
                  <KeyValueGrid
                    items={[
                      {
                        label: "Run",
                        value: <span className="font-mono text-xs">{producerRunId ?? "—"}</span>,
                      },
                      {
                        label: "Execution",
                        value: <span className="font-mono text-xs">{producerExecId ?? "—"}</span>,
                      },
                      {
                        label: "Task",
                        value: <span className="font-mono text-xs">{producerTaskId ?? "—"}</span>,
                      },
                    ]}
                  />
                </OverviewSection>
              )}

              {asset.content_hash && (
                <OverviewSection title="Content hash">
                  <div className="break-all rounded border border-border/70 bg-muted/20 p-2 font-mono text-xs">
                    {asset.content_hash}
                  </div>
                </OverviewSection>
              )}

              {lineage && (lineage.ancestors?.length || lineage.descendants?.length) ? (
                <OverviewSection title="Lineage">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <LineageColumn
                      title="Upstream (ancestors)"
                      nodes={lineage.ancestors ?? []}
                      onSelect={(id) => setSelection({ objectType: "asset", objectId: id })}
                    />
                    <LineageColumn
                      title="Downstream (descendants)"
                      nodes={lineage.descendants ?? []}
                      onSelect={(id) => setSelection({ objectType: "asset", objectId: id })}
                    />
                  </div>
                </OverviewSection>
              ) : null}

              {asset.extra && Object.keys(asset.extra).length > 0 && (
                <OverviewSection title="Kind-specific details">
                  <pre className="overflow-auto border-y border-border/70 bg-muted/20 p-3 font-mono text-xs">
                    {JSON.stringify(asset.extra, null, 2)}
                  </pre>
                </OverviewSection>
              )}
            </OverviewPage>
          ),
        },
        {
          value: "content",
          label: "Content",
          content: <ContentPanel asset={asset} />,
        },
      ]}
    />
  );
};

// Suppress unused-import warning (kept for future virtualised log tail)
void FileText;
void EMPTY_COPY;

export type { AssetKind };
