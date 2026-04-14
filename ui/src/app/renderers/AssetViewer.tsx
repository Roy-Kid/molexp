import { Archive, Download, FileText, Image as ImageIcon, Package } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { DataTableColumn } from "@/app/components/entity";
import {
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityHeader,
  EntityMetric,
  KeyValueGrid,
} from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetFile, ApiAssetResponse, RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const isTextual = (mime: string | undefined, format: string): boolean => {
  if (mime?.startsWith("text/")) return true;
  return ["json", "yaml", "yml", "txt", "md", "markdown", "py", "csv"].includes(
    format.toLowerCase(),
  );
};

const isImage = (mime: string | undefined, format: string): boolean => {
  if (mime?.startsWith("image/")) return true;
  return ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(format.toLowerCase());
};

const ContentPreview = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [textContent, setTextContent] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const textual = isTextual(asset.mimeType, asset.format);
  const image = isImage(asset.mimeType, asset.format);

  useEffect(() => {
    if (!textual && !image) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    fetch(`/api/assets/${encodeURIComponent(asset.assetId)}/download`)
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
        } else if (textual) {
          return blob.text().then((text) => {
            if (cancelled) return;
            setTextContent(text);
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
  }, [asset.assetId, textual, image]);

  const downloadUrl = `/api/assets/${encodeURIComponent(asset.assetId)}/download`;

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
            alt={asset.assetId}
            className="max-h-full max-w-full rounded-md border border-border/60"
          />
        ) : (
          <EmptyState title="Loading…" icon={<ImageIcon className="h-8 w-8" />} />
        )}
      </div>
    );
  }

  if (textual) {
    let displayed = textContent ?? "";
    if (textContent && asset.format.toLowerCase() === "json") {
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
        description={`${formatBytes(asset.size)} · ${asset.format}`}
        icon={<Package className="h-8 w-8" />}
        action={
          <a href={downloadUrl} download={`${asset.assetId}.${asset.format}`}>
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

export const AssetViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const [asset, setAsset] = useState<ApiAssetResponse | null>(null);
  const [notFound, setNotFound] = useState(false);
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  const assetId = selection.objectId;

  useEffect(() => {
    let cancelled = false;
    workspaceApi
      .getAssets()
      .then((all) => {
        if (cancelled) return;
        const match = all.find((a) => a.id === assetId || a.assetId === assetId);
        if (!match) {
          setNotFound(true);
        } else {
          setAsset(match);
          setNotFound(false);
        }
      })
      .catch((err) => {
        console.error("Failed to load asset", err);
        setNotFound(true);
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

  const fileColumns: DataTableColumn<ApiAssetFile>[] = [
    {
      key: "path",
      header: "Path",
      cell: (file) => (
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="font-mono text-xs text-foreground">{file.path}</span>
        </div>
      ),
    },
    {
      key: "size",
      header: "Size",
      width: "w-[120px]",
      cell: (file) => <span className="font-mono text-xs">{formatBytes(file.size)}</span>,
    },
    {
      key: "hash",
      header: "Hash",
      cell: (file) => (
        <span
          className="block max-w-[360px] truncate font-mono text-xs text-muted-foreground"
          title={file.hash}
        >
          {file.hash}
        </span>
      ),
    },
  ];

  const downloadUrl = `/api/assets/${encodeURIComponent(asset.assetId)}/download`;

  const producerRun = asset.producerRunId
    ? snapshot.runs.find((r) => r.id === asset.producerRunId)
    : null;

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        breadcrumbs={breadcrumbs}
        canNavigateUp={canNavigateUp}
        onNavigateUp={navigateUp}
        icon={Archive}
        title={asset.assetId}
        status={assetSummary?.status}
        subtitle={assetSummary?.summary || undefined}
        actions={
          <a href={downloadUrl} download={`${asset.assetId}.${asset.format}`}>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          </a>
        }
        metrics={
          <>
            <EntityMetric label="Files" value={asset.files?.length ?? 0} />
            <EntityMetric label="Size" value={formatBytes(asset.size)} />
          </>
        }
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <Tabs defaultValue="overview" className="flex flex-1 flex-col">
          <div className="border-b border-border/70 bg-muted/10 px-6 py-2 md:px-8">
            <TabsList className="h-auto w-fit justify-start rounded-md bg-transparent p-0">
              <TabsTrigger value="overview" className="rounded-md px-4 py-2 text-sm font-medium">
                Overview
              </TabsTrigger>
              <TabsTrigger value="content" className="rounded-md px-4 py-2 text-sm font-medium">
                Content
              </TabsTrigger>
              <TabsTrigger value="files" className="rounded-md px-4 py-2 text-sm font-medium">
                Files
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent
            value="overview"
            className="m-0 flex flex-1 flex-col overflow-auto p-6 md:p-8"
          >
            <div className="max-w-3xl space-y-6">
              {(assetSummary?.projectId || producerRun) && (
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Relationships
                  </h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {assetSummary?.projectId && (
                      <Button
                        variant="outline"
                        size="sm"
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
                        onClick={() =>
                          setSelection({ objectType: "run", objectId: producerRun.id })
                        }
                      >
                        Producer Run: {producerRun.name || producerRun.id}
                      </Button>
                    )}
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Metadata
                </h3>
                <div className="mt-4">
                  <KeyValueGrid
                    items={[
                      {
                        label: "Asset ID",
                        value: <span className="font-mono text-xs">{asset.assetId}</span>,
                      },
                      {
                        label: "Internal ID",
                        value: <span className="font-mono text-xs">{asset.id}</span>,
                      },
                      { label: "Type", value: asset.type },
                      { label: "Format", value: asset.format },
                      { label: "MIME Type", value: asset.mimeType || "-" },
                      { label: "Size", value: formatBytes(asset.size) },
                      {
                        label: "Content Hash",
                        value: (
                          <span className="break-all font-mono text-xs">{asset.contentHash}</span>
                        ),
                      },
                      { label: "Created", value: new Date(asset.created).toLocaleString() },
                    ]}
                  />
                </div>
              </div>

              {asset.tags && asset.tags.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Tags
                  </h3>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {asset.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="content" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <ContentPreview asset={asset} />
          </TabsContent>

          <TabsContent value="files" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <DataTable
              columns={fileColumns}
              data={asset.files ?? []}
              getRowKey={(file) => file.path}
              empty={
                <EmptyState
                  title={EMPTY_COPY.assets.title}
                  description="This asset has no file manifest."
                />
              }
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
