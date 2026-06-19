import {
  BookText,
  ExternalLink,
  Eye,
  FileText,
  Library as LibraryIcon,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import type { WorkspaceSnapshot } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MarkdownContent } from "@/components/ui/markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AddNoteDialog } from "./AddNoteDialog";
import { AddReferenceDialog } from "./AddReferenceDialog";
import { libraryApi } from "./api";
import { ImportZoteroDialog } from "./ImportZoteroDialog";
import type { LibraryIndex, LibraryNoteEntry, LibraryReference, LibraryScope } from "./types";

// Lazy so monaco-editor (large) is split into an async chunk fetched only
// when a note is actually edited — same pattern as the workspace TextEditor.
const MonacoEditor = lazy(() => import("@monaco-editor/react"));

const WORKSPACE_SCOPE = "__workspace__";

interface LibraryPageProps {
  snapshot: WorkspaceSnapshot;
}

const referenceIds = (ref: LibraryReference): { label: string; href: string }[] => {
  const ids: { label: string; href: string }[] = [];
  if (ref.arxiv)
    ids.push({ label: `arXiv:${ref.arxiv}`, href: `https://arxiv.org/abs/${ref.arxiv}` });
  if (ref.doi) ids.push({ label: `doi:${ref.doi}`, href: `https://doi.org/${ref.doi}` });
  if (ref.url && !ref.arxiv && !ref.doi) ids.push({ label: "link", href: ref.url });
  return ids;
};

const NotesPanel = ({
  index,
  scope,
  onRefresh,
}: {
  index: LibraryIndex;
  scope: LibraryScope;
  onRefresh: () => void;
}): JSX.Element => {
  const [selected, setSelected] = useState<LibraryNoteEntry | null>(index.notes[0] ?? null);
  const [body, setBody] = useState<string>("");
  const [loadingBody, setLoadingBody] = useState(false);
  const [mode, setMode] = useState<"preview" | "edit">("preview");
  const [draft, setDraft] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    // Keep a valid selection as the note set changes.
    if (!selected || !index.notes.some((n) => n.asset_id === selected.asset_id)) {
      setSelected(index.notes[0] ?? null);
    }
  }, [index.notes, selected]);

  useEffect(() => {
    // Reset to preview whenever the selected note changes.
    setMode("preview");
    setSaveError(null);
    if (!selected) {
      setBody("");
      return;
    }
    let cancelled = false;
    setLoadingBody(true);
    libraryApi
      .noteContent(selected.asset_id)
      .then((text) => {
        if (!cancelled) setBody(text);
      })
      .catch((err) => {
        if (!cancelled)
          setBody(`*Failed to load note:* ${err instanceof Error ? err.message : err}`);
      })
      .finally(() => {
        if (!cancelled) setLoadingBody(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const startEdit = (): void => {
    setDraft(body);
    setSaveError(null);
    setMode("edit");
  };

  const handleSave = async (): Promise<void> => {
    if (!selected) return;
    setSaving(true);
    setSaveError(null);
    try {
      await libraryApi.updateNote(scope, selected.asset_id, draft);
      setBody(draft);
      setMode("preview");
      onRefresh();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  if (index.notes.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
        No notes in this scope yet. Use “New note” to capture a decision, spec, or finding.
      </div>
    );
  }

  return (
    <div className="grid h-full grid-cols-[18rem_1fr] overflow-hidden">
      <div className="flex flex-col border-r border-border">
        <ScrollArea className="flex-1">
          <ul className="space-y-0.5 p-2">
            {index.notes.map((note) => {
              const active = selected?.asset_id === note.asset_id;
              return (
                <li key={note.asset_id}>
                  <button
                    type="button"
                    onClick={() => setSelected(note)}
                    className={`flex w-full flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                      active ? "bg-muted" : "hover:bg-muted/50"
                    }`}
                  >
                    <span className="flex items-center gap-1.5 font-medium">
                      <FileText className="h-3.5 w-3.5 text-muted-foreground" />
                      {note.title}
                    </span>
                    {note.summary && (
                      <span className="line-clamp-2 text-xs text-muted-foreground">
                        {note.summary}
                      </span>
                    )}
                    {note.tags.length > 0 && (
                      <span className="flex flex-wrap gap-1 pt-0.5">
                        {note.tags.map((t) => (
                          <Badge key={t} variant="secondary" className="px-1 py-0 text-[10px]">
                            {t}
                          </Badge>
                        ))}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </ScrollArea>
      </div>
      <div className="flex flex-col overflow-hidden">
        {selected && (
          <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2">
            <div className="min-w-0">
              <p className="truncate font-semibold">{selected.title}</p>
              <p className="truncate font-mono text-[11px] text-muted-foreground">
                {selected.path}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {selected.refs.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  cites: {selected.refs.join(", ")}
                </span>
              )}
              {mode === "preview" ? (
                <Button variant="outline" size="sm" onClick={startEdit} disabled={loadingBody}>
                  <Pencil className="mr-1 h-3.5 w-3.5" /> Edit
                </Button>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setMode("preview")}
                    disabled={saving}
                  >
                    <Eye className="mr-1 h-3.5 w-3.5" /> Cancel
                  </Button>
                  <Button size="sm" onClick={() => void handleSave()} disabled={saving}>
                    {saving ? "Saving…" : "Save"}
                  </Button>
                </>
              )}
            </div>
          </div>
        )}
        {saveError && <p className="px-4 py-1 text-sm text-destructive">{saveError}</p>}
        {mode === "edit" ? (
          <div className="min-h-0 flex-1">
            <Suspense
              fallback={<p className="p-4 text-sm text-muted-foreground">Loading editor…</p>}
            >
              <MonacoEditor
                height="100%"
                language="markdown"
                value={draft}
                onChange={(value) => setDraft(value ?? "")}
                options={{
                  minimap: { enabled: false },
                  wordWrap: "on",
                  lineNumbers: "off",
                  fontSize: 13,
                  scrollBeyondLastLine: false,
                }}
              />
            </Suspense>
          </div>
        ) : (
          <ScrollArea className="flex-1 px-4 py-3">
            {loadingBody ? (
              <p className="text-sm text-muted-foreground">Loading…</p>
            ) : (
              <MarkdownContent text={body} />
            )}
          </ScrollArea>
        )}
      </div>
    </div>
  );
};

const ReferencesPanel = ({
  index,
  scope,
  onRefresh,
}: {
  index: LibraryIndex;
  scope: LibraryScope;
  onRefresh: () => void;
}): JSX.Element => {
  const handleDelete = useCallback(
    async (key: string) => {
      try {
        await libraryApi.deleteReference(scope, key);
        onRefresh();
      } catch (err) {
        console.error("Failed to delete reference:", err);
      }
    },
    [scope, onRefresh],
  );

  if (index.references.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
        No references yet. Use “Add reference” to track an arXiv id, DOI, or paper.
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-background">
          <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2 font-medium">Key</th>
            <th className="px-3 py-2 font-medium">Title</th>
            <th className="px-3 py-2 font-medium">Year</th>
            <th className="px-3 py-2 font-medium">Identifiers</th>
            <th className="px-3 py-2 font-medium">Tags</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {index.references.map((ref) => (
            <tr key={ref.key} className="border-b border-border/60 align-top hover:bg-muted/30">
              <td className="px-3 py-2 font-mono text-xs">
                {ref.key}
                {ref.source && (
                  <Badge variant="outline" className="ml-1 px-1 py-0 text-[10px]">
                    {ref.source}
                  </Badge>
                )}
              </td>
              <td className="px-3 py-2">
                <div className="font-medium">{ref.title}</div>
                {ref.authors.length > 0 && (
                  <div className="text-xs text-muted-foreground">{ref.authors.join(", ")}</div>
                )}
                {ref.note && <div className="pt-0.5 text-xs text-muted-foreground">{ref.note}</div>}
              </td>
              <td className="px-3 py-2 tabular-nums">{ref.year ?? "—"}</td>
              <td className="px-3 py-2">
                <div className="flex flex-col gap-1">
                  {referenceIds(ref).map((id) => (
                    <a
                      key={id.label}
                      href={id.href}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 text-primary underline underline-offset-2"
                    >
                      {id.label}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ))}
                  {ref.pdf_path && (
                    <span
                      className="text-muted-foreground"
                      title={`Linked (not copied): ${ref.pdf_path}`}
                    >
                      PDF linked
                    </span>
                  )}
                </div>
              </td>
              <td className="px-3 py-2">
                <div className="flex flex-wrap gap-1">
                  {ref.tags.map((t) => (
                    <Badge key={t} variant="secondary" className="px-1 py-0 text-[10px]">
                      {t}
                    </Badge>
                  ))}
                </div>
              </td>
              <td className="px-3 py-2 text-right">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  aria-label={`Delete ${ref.key}`}
                  onClick={() => void handleDelete(ref.key)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollArea>
  );
};

export const LibraryPage = ({ snapshot }: LibraryPageProps): JSX.Element => {
  const [scopeValue, setScopeValue] = useState<string>(WORKSPACE_SCOPE);
  const [index, setIndex] = useState<LibraryIndex | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addNoteOpen, setAddNoteOpen] = useState(false);
  const [addRefOpen, setAddRefOpen] = useState(false);
  const [importZoteroOpen, setImportZoteroOpen] = useState(false);
  const [tab, setTab] = useState("notes");

  const scope: LibraryScope = useMemo(
    () => (scopeValue === WORKSPACE_SCOPE ? {} : { projectId: scopeValue }),
    [scopeValue],
  );

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    libraryApi
      .getIndex(scope)
      .then(setIndex)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [scope]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <BookText className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold">Library</h1>
            <p className="text-xs text-muted-foreground">
              Notes &amp; literature — previewed here and discoverable by the agent.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Select value={scopeValue} onValueChange={setScopeValue}>
            <SelectTrigger className="h-8 w-56">
              <SelectValue placeholder="Scope" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={WORKSPACE_SCOPE}>Workspace</SelectItem>
              {snapshot.projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  Project · {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={load}
            aria-label="Refresh"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          {tab === "notes" ? (
            <Button size="sm" onClick={() => setAddNoteOpen(true)}>
              <Plus className="mr-1 h-4 w-4" /> New note
            </Button>
          ) : (
            <>
              <Button variant="outline" size="sm" onClick={() => setImportZoteroOpen(true)}>
                <LibraryIcon className="mr-1 h-4 w-4" /> Import Zotero
              </Button>
              <Button size="sm" onClick={() => setAddRefOpen(true)}>
                <Plus className="mr-1 h-4 w-4" /> Add reference
              </Button>
            </>
          )}
        </div>
      </div>

      {error && <p className="px-4 py-2 text-sm text-destructive">{error}</p>}

      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
        <div className="px-4 pt-2">
          <TabsList>
            <TabsTrigger value="notes">Notes ({index?.notes.length ?? 0})</TabsTrigger>
            <TabsTrigger value="references">
              References ({index?.references.length ?? 0})
            </TabsTrigger>
          </TabsList>
        </div>
        <Separator className="mt-2" />
        <div className="min-h-0 flex-1">
          {loading && !index ? (
            <p className="p-4 text-sm text-muted-foreground">Loading library…</p>
          ) : index ? (
            <>
              <TabsContent value="notes" className="m-0 h-full">
                <NotesPanel index={index} scope={scope} onRefresh={load} />
              </TabsContent>
              <TabsContent value="references" className="m-0 h-full">
                <ReferencesPanel index={index} scope={scope} onRefresh={load} />
              </TabsContent>
            </>
          ) : null}
        </div>
      </Tabs>

      <AddNoteDialog
        scope={scope}
        open={addNoteOpen}
        onOpenChange={setAddNoteOpen}
        onCreated={load}
      />
      <AddReferenceDialog
        scope={scope}
        open={addRefOpen}
        onOpenChange={setAddRefOpen}
        onCreated={load}
      />
      <ImportZoteroDialog
        scope={scope}
        open={importZoteroOpen}
        onOpenChange={setImportZoteroOpen}
        onImported={load}
      />
    </div>
  );
};
