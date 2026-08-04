import {
  Archive,
  Download,
  FileJson,
  GitCommitHorizontal,
  Layers,
  Package,
  ScrollText,
  ShieldAlert,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { AssetLineageNode } from "@/api/generated/models/AssetLineageNode";
import type { AssetLineageResponse } from "@/api/generated/models/AssetLineageResponse";
import {
  EntityMetric,
  EntityPage,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { canonicalStatusFor } from "@/app/components/entity/status";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, AssetKind, RendererProps } from "@/app/types";
import { Code as InlineCode } from "@/components/ui/code";
import {
  WorkbenchAction,
  WorkbenchIconAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
  WorkbenchTag,
} from "@/components/workbench";
import { formatDateTime } from "@/lib/datetime";
import { formatBytes } from "@/lib/format-bytes";

// ── Helpers ────────────────────────────────────────────────────────────────

const KIND_META: Record<string, { label: string; icon: typeof Archive; iconClassName: string }> = {
  data: { label: "Data", icon: Package, iconClassName: "text-muted-foreground" },
  artifact: { label: "Artifact", icon: Archive, iconClassName: "text-muted-foreground" },
  log: { label: "Log", icon: ScrollText, iconClassName: "text-muted-foreground" },
  checkpoint: {
    label: "Checkpoint",
    icon: GitCommitHorizontal,
    iconClassName: "text-muted-foreground",
  },
  error_trace: { label: "Error Trace", icon: ShieldAlert, iconClassName: "text-muted-foreground" },
  execution_state: {
    label: "Execution State",
    icon: Layers,
    iconClassName: "text-muted-foreground",
  },
  output: { label: "Output", icon: FileJson, iconClassName: "text-muted-foreground" },
};

const kindMeta = (kind: string) =>
  KIND_META[kind] ?? { label: kind, icon: Archive, iconClassName: "text-muted-foreground" };

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
  const mime = extraValue<string>(asset, "mime");
  const size = extraValue<number>(asset, "size") ?? null;
  const textual = isTextual(mime, asset.path);
  const image = isImage(mime, asset.path);
  const [textContent, setTextContent] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(textual || image);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    if (!textual && !image) {
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;
    setLoading(true);
    setError(null);
    setTextContent(null);
    setImageUrl(null);

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
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset.id, textual, image, requestVersion]);

  const downloadUrl = `/api/assets/${encodeURIComponent(asset.id)}/content`;

  if (error) {
    return (
      <WorkbenchOperationState
        kind="error"
        title="Could not load asset content"
        detail={error}
        action={
          <WorkbenchRetryAction onClick={() => setRequestVersion((version) => version + 1)} />
        }
      />
    );
  }

  if (image) {
    return (
      <div className="flex h-full items-center justify-center bg-muted/20 p-6">
        {loading ? (
          <WorkbenchOperationState kind="loading" title="Loading image…" />
        ) : imageUrl ? (
          <img
            src={imageUrl}
            alt={asset.name}
            className="max-h-full max-w-full rounded-control border border-border/60"
          />
        ) : (
          <WorkbenchOperationState
            kind="empty"
            title="Image content is empty"
            detail="The asset loaded successfully but did not return a preview."
          />
        )}
      </div>
    );
  }

  if (textual) {
    if (loading) {
      return <WorkbenchOperationState kind="loading" title="Loading asset content…" />;
    }
    if (textContent === "") {
      return (
        <WorkbenchOperationState
          kind="empty"
          title="Asset content is empty"
          detail="The asset loaded successfully but contains no text."
        />
      );
    }
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
        <pre className="whitespace-pre-wrap break-words px-6 py-4 font-mono text-label text-foreground">
          {displayed}
        </pre>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center">
      <WorkbenchOperationState
        kind="empty"
        title="Binary preview unavailable"
        detail={`${formatBytes(size)} · ${mime ?? "unknown"}`}
        action={
          <WorkbenchIconAction label={`Download ${asset.name}`} asChild>
            <a href={downloadUrl} download={asset.name}>
              <Download className="h-4 w-4" />
            </a>
          </WorkbenchIconAction>
        }
      />
    </div>
  );
};

const LogTail = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [lines, setLines] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    let cancelled = false;
    setLines(null);
    setLoading(true);
    setError(null);
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
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [asset.id, requestVersion]);

  if (error) {
    return (
      <WorkbenchOperationState
        kind="error"
        title="Could not load log"
        detail={error}
        action={
          <WorkbenchRetryAction onClick={() => setRequestVersion((version) => version + 1)} />
        }
      />
    );
  }

  if (loading) {
    return <WorkbenchOperationState kind="loading" title="Loading log…" skeletonRows={5} />;
  }

  if (!lines) {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="No log output"
        detail="The log loaded successfully but contains no lines."
      />
    );
  }

  return (
    <div className="h-full overflow-auto bg-canvas px-4 py-3 font-mono text-label text-foreground">
      <pre className="whitespace-pre-wrap break-words">{lines}</pre>
    </div>
  );
};

const JsonPreview = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [text, setText] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    let cancelled = false;
    setText(null);
    setLoading(true);
    setError(null);
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
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [asset.id, requestVersion]);

  if (error) {
    return (
      <WorkbenchOperationState
        kind="error"
        title="Could not load structured content"
        detail={error}
        action={
          <WorkbenchRetryAction onClick={() => setRequestVersion((version) => version + 1)} />
        }
      />
    );
  }

  if (loading) {
    return <WorkbenchOperationState kind="loading" title="Loading structured content…" />;
  }

  if (text === "") {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="Structured content is empty"
        detail="The asset loaded successfully but contains no content."
      />
    );
  }

  return (
    <div className="h-full overflow-auto">
      <pre className="whitespace-pre-wrap break-words px-6 py-4 font-mono text-label text-foreground">
        {text}
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
      <div className="border-b border-status-failed/20 bg-status-failed-soft px-6 py-4">
        <div className="text-label font-semibold uppercase text-status-failed-foreground">
          {exceptionType ?? "Unknown exception"}
        </div>
        <div className="mt-1 text-body-lg text-foreground">{message ?? "(no message)"}</div>
        {executionId && (
          <div className="mt-1 font-mono text-label text-muted-foreground">
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
      <div className="space-y-2 text-label text-muted-foreground">
        <div className="mb-1 font-semibold text-foreground">{title}</div>
        <div className="border-y border-border/60 py-2">—</div>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="mb-2 text-label font-semibold text-foreground">
        {title} <span className="text-muted-foreground">({nodes.length})</span>
      </div>
      <ul className="divide-y divide-border/60 border-y border-border/60">
        {nodes.map((node) => {
          const meta = kindMeta(node.kind);
          const Icon = meta.icon;
          return (
            <li key={node.id}>
              <WorkbenchAction
                kind="ghost"
                size="content"
                type="button"
                onClick={() => onSelect(node.id)}
                className="flex w-full items-center gap-2 px-2 py-2 text-left text-label hover:bg-interactive"
              >
                <Icon className={`h-3.5 w-3.5 shrink-0 ${meta.iconClassName}`} />
                <span className="flex-1 truncate font-mono">{node.name}</span>
                <WorkbenchTag meaning="metadata" className="text-micro">
                  {meta.label}
                </WorkbenchTag>
              </WorkbenchAction>
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
  const [assetLoading, setAssetLoading] = useState(true);
  const [assetError, setAssetError] = useState<string | null>(null);
  const [settledAssetId, setSettledAssetId] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [assetRequestVersion, setAssetRequestVersion] = useState(0);
  const [lineage, setLineage] = useState<AssetLineageResponse | null>(null);
  const [lineageLoading, setLineageLoading] = useState(true);
  const [lineageError, setLineageError] = useState<string | null>(null);
  const [lineageRequestVersion, setLineageRequestVersion] = useState(0);
  const { setSelection } = useNavigationState(snapshot);

  const assetId = selection.objectId;

  useEffect(() => {
    void assetRequestVersion;
    let cancelled = false;
    setAsset(null);
    setAssetLoading(true);
    setAssetError(null);
    setNotFound(false);
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
        if (cancelled) return;
        setAssetError(err instanceof Error ? err.message : "Failed to load asset");
      })
      .finally(() => {
        if (!cancelled) {
          setAssetLoading(false);
          setSettledAssetId(assetId);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [assetId, assetRequestVersion]);

  useEffect(() => {
    void lineageRequestVersion;
    let cancelled = false;
    setLineage(null);
    setLineageError(null);
    if (!assetId) {
      setLineageLoading(false);
      return;
    }
    setLineageLoading(true);
    workspaceApi
      .getAssetLineage(assetId)
      .then((res) => {
        if (cancelled) return;
        setLineage(res);
      })
      .catch((err) => {
        if (cancelled) return;
        setLineageError(err instanceof Error ? err.message : "Failed to load asset lineage");
      })
      .finally(() => {
        if (!cancelled) setLineageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [assetId, lineageRequestVersion]);

  const assetSummary = useMemo(
    () => snapshot.assets.find((a) => a.id === assetId),
    [snapshot.assets, assetId],
  );

  if (assetLoading || settledAssetId !== assetId) {
    return (
      <WorkbenchOperationState
        kind="loading"
        title="Loading asset…"
        skeletonRows={6}
        className="h-full"
      />
    );
  }

  if (assetError) {
    return (
      <WorkbenchOperationState
        kind="error"
        title="Could not load asset"
        detail={assetError}
        action={
          <WorkbenchRetryAction
            onClick={() => {
              setAssetLoading(true);
              setAssetRequestVersion((version) => version + 1);
            }}
          />
        }
      />
    );
  }

  if (notFound || !asset) {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="Asset not found"
        detail="The asset list loaded successfully, but this asset is no longer present."
      />
    );
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
      icon={meta.icon}
      title={asset.name}
      status={assetSummary?.status}
      subtitle={`${meta.label} · ${scopeLabel}`}
      actions={
        <WorkbenchIconAction label={`Download ${asset.name}`} asChild>
          <a href={downloadUrl} download={asset.name}>
            <Download className="h-4 w-4" />
          </a>
        </WorkbenchIconAction>
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
                      <OverviewHighlight
                        label="Status"
                        value={canonicalStatusFor(assetSummary?.status) ?? "unknown"}
                      />
                    </OverviewHighlightGrid>
                  </OverviewSection>

                  {(assetSummary?.projectId || producerRun) && (
                    <OverviewSection title="Relationships">
                      <div className="flex flex-wrap gap-2">
                        {assetSummary?.projectId && (
                          <WorkbenchAction
                            kind="secondary"
                            size="compact"
                            className="h-control-compact px-2 text-label"
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
                          </WorkbenchAction>
                        )}
                        {producerRun && (
                          <WorkbenchAction
                            kind="secondary"
                            size="compact"
                            className="h-control-compact px-2 text-label"
                            onClick={() =>
                              setSelection({ objectType: "run", objectId: producerRun.id })
                            }
                          >
                            Producer Run: {producerRun.name || producerRun.id}
                          </WorkbenchAction>
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
                      <div className="flex flex-wrap gap-2">
                        {tagEntries.map(([key, value]) => (
                          <WorkbenchTag key={key} className="text-label">
                            {key}: {value}
                          </WorkbenchTag>
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
                      value: <span className="font-mono text-label">{asset.id}</span>,
                    },
                    { label: "Name", value: asset.name },
                    { label: "Kind", value: meta.label },
                    {
                      label: "Scope",
                      value: <span className="font-mono text-label">{scopeLabel}</span>,
                    },
                    {
                      label: "Path",
                      value: <span className="break-all font-mono text-label">{asset.path}</span>,
                    },
                    {
                      label: "Created",
                      value: (
                        <span title={asset.created_at}>{formatDateTime(asset.created_at)}</span>
                      ),
                    },
                    {
                      label: "Updated",
                      value: (
                        <span title={asset.updated_at}>{formatDateTime(asset.updated_at)}</span>
                      ),
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
                        value: <span className="font-mono text-label">{producerRunId ?? "—"}</span>,
                      },
                      {
                        label: "Execution",
                        value: (
                          <span className="font-mono text-label">{producerExecId ?? "—"}</span>
                        ),
                      },
                      {
                        label: "Task",
                        value: (
                          <span className="font-mono text-label">{producerTaskId ?? "—"}</span>
                        ),
                      },
                    ]}
                  />
                </OverviewSection>
              )}

              {asset.content_hash && (
                <OverviewSection title="Content hash">
                  <InlineCode className="block break-all border-y border-border/60 py-2 font-mono text-label">
                    {asset.content_hash}
                  </InlineCode>
                </OverviewSection>
              )}

              <OverviewSection title="Lineage">
                {lineageLoading ? (
                  <WorkbenchOperationState
                    kind="loading"
                    density="compact"
                    title="Loading lineage…"
                    skeletonRows={2}
                  />
                ) : lineageError ? (
                  <WorkbenchOperationState
                    kind="error"
                    density="compact"
                    title="Could not load lineage"
                    detail={lineageError}
                    action={
                      <WorkbenchRetryAction
                        onClick={() => setLineageRequestVersion((version) => version + 1)}
                      />
                    }
                  />
                ) : lineage && (lineage.ancestors?.length || lineage.descendants?.length) ? (
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
                ) : (
                  <WorkbenchOperationState
                    kind="empty"
                    density="compact"
                    title="No lineage recorded"
                    detail="This asset has no known upstream or downstream assets."
                  />
                )}
              </OverviewSection>

              {asset.extra && Object.keys(asset.extra).length > 0 && (
                <OverviewSection title="Kind-specific details">
                  <pre className="overflow-auto border-y border-border/70 bg-muted/20 p-3 font-mono text-label">
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
          content: <ContentPanel key={asset.id} asset={asset} />,
        },
      ]}
    />
  );
};

export type { AssetKind };
