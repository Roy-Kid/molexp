import "katex/dist/katex.min.css";
import { defaultValueCtx, Editor, rootCtx } from "@milkdown/kit/core";
import { listener, listenerCtx } from "@milkdown/kit/plugin/listener";
import { commonmark } from "@milkdown/kit/preset/commonmark";
import { gfm } from "@milkdown/kit/preset/gfm";
import { math } from "@milkdown/plugin-math";
import { Milkdown, MilkdownProvider, useEditor } from "@milkdown/react";
import { Code2, Eye, Save } from "lucide-react";
import { type JSX, lazy, Suspense, useCallback, useMemo, useState } from "react";
import type { NoteDetailResponse } from "@/api/generated/models/NoteDetailResponse";
import { workspaceApi } from "@/app/state/api";
import type { WorkspaceSnapshot } from "@/app/types";
import { Button } from "@/components/ui/button";
import { buildNoteDocUpdate, isDirty } from "./noteDraft";
import { SlashMenu } from "./SlashMenu";

// Monaco is a heavy dependency; lazy-load it so the source-mode chunk is only
// fetched when the user toggles into it (mirrors ``plugins/editor/TextEditor``).
const MonacoEditor = lazy(() => import("@monaco-editor/react"));

type EditMode = "wysiwyg" | "source";

// ProseMirror-density styling for the Milkdown editing surface. The project
// ships no Tailwind typography plugin, so element rhythm is expressed through
// descendant selectors against theme tokens (same approach as MarkdownContent).
const MILKDOWN_HOST = [
  "h-full",
  "[&_.ProseMirror]:min-h-full [&_.ProseMirror]:px-4 [&_.ProseMirror]:py-3",
  "[&_.ProseMirror]:text-sm [&_.ProseMirror]:leading-relaxed [&_.ProseMirror]:text-foreground",
  "[&_.ProseMirror]:outline-none",
  "[&_.ProseMirror_h1]:mt-4 [&_.ProseMirror_h1]:mb-1.5 [&_.ProseMirror_h1]:text-base [&_.ProseMirror_h1]:font-semibold",
  "[&_.ProseMirror_h2]:mt-3.5 [&_.ProseMirror_h2]:mb-1 [&_.ProseMirror_h2]:text-[0.9375rem] [&_.ProseMirror_h2]:font-semibold",
  "[&_.ProseMirror_h3]:mt-3 [&_.ProseMirror_h3]:mb-1 [&_.ProseMirror_h3]:text-sm [&_.ProseMirror_h3]:font-semibold",
  "[&_.ProseMirror_p]:my-2",
  "[&_.ProseMirror_ul]:my-2 [&_.ProseMirror_ul]:list-disc [&_.ProseMirror_ul]:pl-5",
  "[&_.ProseMirror_ol]:my-2 [&_.ProseMirror_ol]:list-decimal [&_.ProseMirror_ol]:pl-5",
  "[&_.ProseMirror_a]:text-primary [&_.ProseMirror_a]:underline [&_.ProseMirror_a]:underline-offset-2",
  "[&_.ProseMirror_blockquote]:my-2 [&_.ProseMirror_blockquote]:border-l-2 [&_.ProseMirror_blockquote]:border-border [&_.ProseMirror_blockquote]:pl-3 [&_.ProseMirror_blockquote]:text-muted-foreground",
  "[&_.ProseMirror_code]:rounded-sm [&_.ProseMirror_code]:bg-muted [&_.ProseMirror_code]:px-1 [&_.ProseMirror_code]:py-px [&_.ProseMirror_code]:font-mono [&_.ProseMirror_code]:text-[0.85em]",
  "[&_.ProseMirror_pre]:my-2 [&_.ProseMirror_pre]:rounded-md [&_.ProseMirror_pre]:border [&_.ProseMirror_pre]:border-border/60 [&_.ProseMirror_pre]:bg-muted/50 [&_.ProseMirror_pre]:px-3 [&_.ProseMirror_pre]:py-2.5 [&_.ProseMirror_pre]:font-mono [&_.ProseMirror_pre]:text-xs",
].join(" ");

/**
 * Milkdown (ProseMirror, markdown-native) WYSIWYG surface. Seeded from
 * ``initialValue`` and streams serialized markdown back through ``onChange``.
 * Remount it (via a ``key`` at the call site) to reseed the document after a
 * source-mode edit — Milkdown does not accept controlled value updates.
 */
const MilkdownSurface = ({
  initialValue,
  onChange,
}: {
  initialValue: string;
  onChange: (markdown: string) => void;
}): JSX.Element => {
  useEditor((root) =>
    Editor.make()
      .config((ctx) => {
        ctx.set(rootCtx, root);
        ctx.set(defaultValueCtx, initialValue);
        ctx.get(listenerCtx).markdownUpdated((_ctx, markdown) => onChange(markdown));
      })
      .use(commonmark)
      .use(gfm)
      .use(math)
      .use(listener),
  );
  return <Milkdown />;
};

/**
 * Editable view of a Note body. A Milkdown WYSIWYG editor and a Monaco source
 * editor share one markdown string; ``index.md`` stays the single source of
 * truth (we serialize to markdown, never block-JSON). Save is gated on the
 * normalized dirty state, routes through ``workspaceApi.updateNoteDoc`` (the
 * generated client), and surfaces failures as an inline error bar — the repo
 * ships no toast library.
 */
export const NoteEditor = ({
  note,
  onSaved,
  snapshot,
  onEmbedded,
}: {
  note: NoteDetailResponse;
  onSaved: (updated: NoteDetailResponse) => void;
  /** Workspace entities the "/" menu can embed; enables the SlashMenu when set. */
  snapshot?: WorkspaceSnapshot;
  /** Fired after an embed edge is written so the host can refetch the cards. */
  onEmbedded?: () => void;
}): JSX.Element => {
  const [mode, setMode] = useState<EditMode>("wysiwyg");
  const [markdown, setMarkdown] = useState<string>(note.body);
  // Bumped whenever we re-enter WYSIWYG so the ProseMirror doc reseeds from the
  // latest markdown (e.g. after a source-mode edit).
  const [surfaceKey, setSurfaceKey] = useState<number>(0);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = useMemo(() => isDirty(note.body, markdown), [note.body, markdown]);

  const enterSource = useCallback(() => setMode("source"), []);
  const enterWysiwyg = useCallback(() => {
    setSurfaceKey((key) => key + 1);
    setMode("wysiwyg");
  }, []);

  // Drop a slash-menu block at the end of the doc, then reseed the WYSIWYG
  // surface so the inserted markdown renders (Milkdown takes no controlled
  // value). `index.md` stays the source of truth — we only append markdown.
  const insertMarkdown = useCallback((snippet: string): void => {
    if (!snippet) return;
    setMarkdown(
      (prev) => `${prev}${prev.length > 0 && !prev.endsWith("\n") ? "\n" : ""}${snippet}`,
    );
    setSurfaceKey((key) => key + 1);
  }, []);

  const handleSave = useCallback(async () => {
    if (!dirty || saving) return;
    setSaving(true);
    setError(null);
    try {
      const update = buildNoteDocUpdate(note.relPath, markdown);
      const persisted = await workspaceApi.updateNoteDoc(update.path, update.body);
      onSaved(persisted);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save note.");
    } finally {
      setSaving(false);
    }
  }, [dirty, saving, note.relPath, markdown, onSaved]);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex flex-none items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-md border border-border/60 bg-muted/30 p-0.5">
            <Button
              type="button"
              variant={mode === "wysiwyg" ? "secondary" : "ghost"}
              size="sm"
              className="gap-1.5"
              onClick={enterWysiwyg}
            >
              <Eye className="h-4 w-4" /> Editor
            </Button>
            <Button
              type="button"
              variant={mode === "source" ? "secondary" : "ghost"}
              size="sm"
              className="gap-1.5"
              onClick={enterSource}
            >
              <Code2 className="h-4 w-4" /> Source
            </Button>
          </div>
          {snapshot && (
            <SlashMenu
              notePath={note.relPath}
              snapshot={snapshot}
              onInsert={insertMarkdown}
              onEmbedded={onEmbedded}
            />
          )}
        </div>
        <div className="flex items-center gap-2">
          {dirty && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
          <Button
            type="button"
            size="sm"
            className="gap-1.5"
            onClick={handleSave}
            disabled={!dirty || saving}
          >
            <Save className="h-4 w-4" />
            {saving ? "Saving…" : "Save"}
          </Button>
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="flex-none rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
        >
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto rounded-md border border-border/60 bg-card">
        {mode === "wysiwyg" ? (
          <MilkdownProvider>
            <div className={MILKDOWN_HOST}>
              <MilkdownSurface key={surfaceKey} initialValue={markdown} onChange={setMarkdown} />
            </div>
          </MilkdownProvider>
        ) : (
          <Suspense
            fallback={<div className="p-4 text-sm text-muted-foreground">Loading editor…</div>}
          >
            <MonacoEditor
              height="100%"
              language="markdown"
              value={markdown}
              theme="light"
              onChange={(next) => setMarkdown(next ?? "")}
              options={{
                minimap: { enabled: false },
                wordWrap: "on",
                scrollBeyondLastLine: false,
              }}
            />
          </Suspense>
        )}
      </div>
    </div>
  );
};
