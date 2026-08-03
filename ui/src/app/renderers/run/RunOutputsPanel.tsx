/**
 * Core Run **Outputs** tab — inventory of run products + plugin-routed preview.
 *
 * Science viewers (molplot / molvis / …) attach via FilePreview plugins and
 * FileType contributions; this panel never imports those packages.
 */

import { FileCode2, FileJson, FileQuestion, Image as ImageIcon, Package } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/app/components/entity";
import type { ApiAssetResponse } from "@/app/types";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import { cn } from "@/lib/utils";

export type OutputFilter = "all" | "science" | "figures" | "source" | "other";

export interface RunResultEntry {
  key: string;
  value: unknown;
}

export interface RunOutputsPanelProps {
  assets: ApiAssetResponse[];
  results: RunResultEntry[];
  loading?: boolean;
}

const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]);
const TEXT_EXT = new Set([".py", ".txt", ".json", ".md", ".csv", ".yml", ".yaml", ".log"]);

const extOf = (name: string): string => {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
};

/** Prefer basename of path — asset.name may drop the extension (e.g. error_exec-…). */
const basenameOf = (asset: ApiAssetResponse): string => {
  const p = asset.path?.replace(/\\/g, "/") ?? "";
  const base = p.includes("/") ? p.slice(p.lastIndexOf("/") + 1) : p;
  return base || asset.name;
};

const extOfAsset = (asset: ApiAssetResponse): string => {
  const fromName = extOf(asset.name);
  if (fromName) return fromName;
  return extOf(basenameOf(asset));
};

const isImage = (asset: ApiAssetResponse): boolean => IMAGE_EXT.has(extOfAsset(asset));

const isTextish = (asset: ApiAssetResponse): boolean => {
  if (TEXT_EXT.has(extOfAsset(asset))) return true;
  // Lifecycle error traces / logs often use names without a suffix.
  const kind = (asset as { kind?: string }).kind;
  if (kind === "error_trace" || kind === "log") return true;
  const mime = (asset as { mime?: string }).mime ?? "";
  return mime.startsWith("text/") || mime === "application/json";
};

const isSource = (asset: ApiAssetResponse): boolean =>
  asset.tags?.role === "source" ||
  asset.path.includes("/source/") ||
  asset.name.startsWith("source/");

const isScience = (asset: ApiAssetResponse): boolean =>
  asset.tags?.molrec === "true" ||
  Boolean(asset.tags?.molrec_sections) ||
  asset.name.includes(".molexp-artifact.json");

type Product =
  | { kind: "asset"; id: string; asset: ApiAssetResponse; label: string; filter: OutputFilter }
  | { kind: "result"; id: string; key: string; value: unknown; label: string; filter: "other" };

const productFilter = (p: Product): OutputFilter => {
  if (p.kind === "result") return "other";
  if (isScience(p.asset)) return "science";
  if (isSource(p.asset)) return "source";
  if (isImage(p.asset)) return "figures";
  return "other";
};

const buildProducts = (assets: ApiAssetResponse[], results: RunResultEntry[]): Product[] => {
  const fromAssets: Product[] = assets.map((asset) => {
    const base: Product = {
      kind: "asset",
      id: `asset:${asset.id}`,
      asset,
      label: asset.name,
      filter: "other",
    };
    return { ...base, filter: productFilter(base) };
  });
  const fromResults: Product[] = results.map((r) => ({
    kind: "result" as const,
    id: `result:${r.key}`,
    key: r.key,
    value: r.value,
    label: r.key,
    filter: "other" as const,
  }));
  return [...fromAssets, ...fromResults];
};

const FILTERS: { id: OutputFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "science", label: "Science" },
  { id: "figures", label: "Figures" },
  { id: "source", label: "Source" },
  { id: "other", label: "Other" },
];

const ProductIcon = ({ product }: { product: Product }): JSX.Element => {
  if (product.kind === "result") return <FileJson className="h-4 w-4 text-muted-foreground" />;
  if (isScience(product.asset)) return <Package className="h-4 w-4 text-status-running" />;
  if (isSource(product.asset)) return <FileCode2 className="h-4 w-4 text-muted-foreground" />;
  if (isImage(product.asset)) return <ImageIcon className="h-4 w-4 text-info" />;
  return <FileQuestion className="h-4 w-4 text-muted-foreground" />;
};

const AssetPreview = ({ asset }: { asset: ApiAssetResponse }): JSX.Element => {
  const [text, setText] = useState<string | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const hasSidecar =
    Boolean(asset.has_preview_sidecar) ||
    Boolean((asset as { hasPreviewSidecar?: boolean }).hasPreviewSidecar);
  const plugin = filePreviewPluginRegistry.getPluginForFile(asset.name, asset.path, {
    hasPreviewSidecar: hasSidecar,
  });

  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;
    setError(null);
    setText(null);
    setBlobUrl(null);

    if (isImage(asset)) {
      fetch(`/api/assets/${encodeURIComponent(asset.id)}/content`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.blob();
        })
        .then((blob) => {
          if (cancelled) return;
          url = URL.createObjectURL(blob);
          setBlobUrl(url);
        })
        .catch((err) => {
          if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load");
        });
      return () => {
        cancelled = true;
        if (url) URL.revokeObjectURL(url);
      };
    }

    // Text-ish files for core fallback (path/kind, not only display name).
    if (isTextish(asset)) {
      fetch(`/api/assets/${encodeURIComponent(asset.id)}/content`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.text();
        })
        .then((body) => {
          if (!cancelled) setText(body.slice(0, 200_000));
        })
        .catch((err) => {
          if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load");
        });
    }

    return () => {
      cancelled = true;
    };
  }, [asset]);

  if (asset.tags?.molrec === "true") {
    const sections = asset.tags.molrec_sections ?? "(sections unknown)";
    return (
      <div className="space-y-2 p-4 text-sm">
        <p className="font-medium text-foreground">MolRec record</p>
        <p className="text-muted-foreground">
          Sections: <span className="font-mono text-xs">{sections}</span>
        </p>
        <p className="text-label text-muted-foreground">
          Open with the molvis (frame/trajectory) or molplot (observables) plugin when those
          sections are present — core only lists the product.
        </p>
        <a
          className="text-label text-primary underline"
          href={`/api/assets/${encodeURIComponent(asset.id)}/content`}
        >
          Download asset
        </a>
      </div>
    );
  }

  if (plugin && !isImage(asset)) {
    // Plugin components expect file content props; for asset-backed previews we
    // only use plugins that can work with name/path (sidecars). Otherwise fall through.
    const Plugin = plugin.Component;
    return (
      <div className="h-full min-h-[12rem] overflow-auto p-2">
        <Plugin content="" name={asset.name} path={asset.path} folderId="" assetId={asset.id} />
      </div>
    );
  }

  if (error) {
    return <p className="p-4 text-sm text-status-failed-foreground">{error}</p>;
  }

  if (blobUrl) {
    return (
      <div className="flex h-full items-center justify-center overflow-auto p-4">
        <img src={blobUrl} alt={asset.name} className="max-h-full max-w-full object-contain" />
      </div>
    );
  }

  if (text !== null) {
    return (
      <pre className="h-full overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-micro text-foreground">
        {text}
      </pre>
    );
  }

  return (
    <div className="flex h-full flex-col items-start justify-center gap-2 p-4 text-sm text-muted-foreground">
      <p>No inline preview for this product.</p>
      <a
        className="text-primary underline"
        href={`/api/assets/${encodeURIComponent(asset.id)}/content`}
      >
        Download {asset.name}
      </a>
    </div>
  );
};

export const RunOutputsPanel = ({
  assets,
  results,
  loading = false,
}: RunOutputsPanelProps): JSX.Element => {
  const products = useMemo(() => buildProducts(assets, results), [assets, results]);
  const [filter, setFilter] = useState<OutputFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const visible = useMemo(
    () => (filter === "all" ? products : products.filter((p) => p.filter === filter)),
    [products, filter],
  );

  useEffect(() => {
    if (visible.length === 0) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !visible.some((p) => p.id === selectedId)) {
      setSelectedId(visible[0].id);
    }
  }, [visible, selectedId]);

  const selected = visible.find((p) => p.id === selectedId) ?? null;

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Loading outputs…
      </div>
    );
  }

  if (products.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <EmptyState
          icon={<Package className="h-5 w-5" />}
          title="No outputs landed"
          description="Register artifacts (prefer MolRec from molpy) and source via builtin run_land, or finish a workflow that saves products under this run."
        />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="flex flex-none items-center gap-1 border-b border-border px-2 py-2">
        {FILTERS.map((f) => {
          const count =
            f.id === "all" ? products.length : products.filter((p) => p.filter === f.id).length;
          if (f.id !== "all" && count === 0) return null;
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              className={cn(
                "rounded-[var(--radius-control)] px-2 py-1 text-label font-medium transition-colors",
                filter === f.id
                  ? "bg-interactive text-foreground"
                  : "text-muted-foreground hover:bg-interactive/60 hover:text-foreground",
              )}
            >
              {f.label}
              <span className="ml-1 tabular-nums text-muted-foreground">{count}</span>
            </button>
          );
        })}
      </div>

      <div className="flex min-h-0 flex-1">
        <ul className="w-56 flex-none space-y-1 overflow-auto border-r border-border p-2">
          {visible.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                onClick={() => setSelectedId(p.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-[var(--radius-control)] px-2 py-2 text-left text-label transition-colors",
                  selectedId === p.id
                    ? "bg-interactive text-foreground"
                    : "text-muted-foreground hover:bg-interactive/50 hover:text-foreground",
                )}
              >
                <ProductIcon product={p} />
                <span className="min-w-0 flex-1 truncate font-mono text-micro">{p.label}</span>
              </button>
            </li>
          ))}
        </ul>

        <div className="min-w-0 flex-1 overflow-auto">
          {!selected ? null : selected.kind === "result" ? (
            <pre className="whitespace-pre-wrap break-words p-4 font-mono text-micro">
              {JSON.stringify(selected.value, null, 2)}
            </pre>
          ) : (
            <AssetPreview asset={selected.asset} />
          )}
        </div>
      </div>
    </div>
  );
};
