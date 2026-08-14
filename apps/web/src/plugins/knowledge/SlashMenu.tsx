import {
  Archive,
  ChevronLeft,
  Code2,
  FlaskConical,
  Heading1,
  Heading2,
  Heading3,
  Link2,
  List,
  ListOrdered,
  ListTodo,
  Loader2,
  Minus,
  PlayCircle,
  Quote,
  Slash,
  Table,
} from "lucide-react";
import { type ComponentType, type JSX, type ReactNode, useState } from "react";
import type { EmbedResponse } from "@/api/generated/models/EmbedResponse";
import { type EmbedTargetKind, workspaceApi } from "@/app/state/api";
import type { WorkspaceSnapshot } from "@/app/types";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { WorkbenchIconAction } from "@/components/workbench";
import { cn } from "@/lib/utils";
import { SLASH_COMMANDS, slashCommandMarkdown } from "./slashCommands";

const BLOCK_ICON: Record<string, ComponentType<{ className?: string }>> = {
  heading1: Heading1,
  heading2: Heading2,
  heading3: Heading3,
  bulletList: List,
  orderedList: ListOrdered,
  checkbox: ListTodo,
  quote: Quote,
  divider: Minus,
  codeBlock: Code2,
  table: Table,
};

interface EmbedGroup {
  kind: EmbedTargetKind;
  label: string;
  icon: ComponentType<{ className?: string }>;
  items: Array<{ id: string; label: string }>;
}

const buildEmbedGroups = (snapshot: WorkspaceSnapshot): EmbedGroup[] =>
  [
    {
      kind: "experiment" as const,
      label: "Experiments",
      icon: FlaskConical,
      items: snapshot.experiments.map((e) => ({ id: e.id, label: e.name })),
    },
    {
      kind: "run" as const,
      label: "Runs",
      icon: PlayCircle,
      items: snapshot.runs.map((r) => ({ id: r.id, label: r.name || r.id })),
    },
    {
      kind: "asset" as const,
      label: "Assets",
      icon: Archive,
      items: snapshot.assets.map((a) => ({ id: a.id, label: a.name })),
    },
  ].filter((group) => group.items.length > 0);

interface SlashMenuProps {
  /** The source note's bundle-relative path (embed edge origin). */
  notePath: string;
  /** Workspace entities that can be embedded. */
  snapshot: WorkspaceSnapshot;
  /** Called with the markdown snippet to drop at the cursor for a block insert. */
  onInsert: (markdown: string) => void;
  /** Called after an embed edge is written, so the host can refetch the cards. */
  onEmbedded?: (response: EmbedResponse) => void;
  /** Override the default trigger button. */
  trigger?: ReactNode;
}

/**
 * The Notion-style "/" menu — a popover command palette listing block inserts
 * (driven by the pure `slashCommands` map) plus an "Embed entity" action that
 * writes one typed provenance edge through `workspaceApi.embedEntity`.
 *
 * The interactive wiring (Milkdown keybinding, cursor insertion) is
 * UI-verification only; the binding unit is `slashCommands`. Blocks insert
 * markdown so `index.md` stays the single source of truth — never block-JSON.
 */
export const SlashMenu = ({
  notePath,
  snapshot,
  onInsert,
  onEmbedded,
  trigger,
}: SlashMenuProps): JSX.Element => {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<"blocks" | "embed">("blocks");
  const [search, setSearch] = useState("");
  const [embedding, setEmbedding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = (): void => {
    setPage("blocks");
    setSearch("");
    setError(null);
  };

  const handleOpenChange = (next: boolean): void => {
    setOpen(next);
    if (!next) reset();
  };

  const insertBlock = (id: string): void => {
    onInsert(slashCommandMarkdown(id));
    handleOpenChange(false);
  };

  const embed = async (targetKind: EmbedTargetKind, target: string): Promise<void> => {
    setEmbedding(true);
    setError(null);
    try {
      const response = await workspaceApi.embedEntity(notePath, { targetKind, target });
      onEmbedded?.(response);
      handleOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to embed entity.");
    } finally {
      setEmbedding(false);
    }
  };

  const embedGroups = buildEmbedGroups(snapshot);

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        {trigger ?? (
          <WorkbenchIconAction label="Insert block">
            <Slash className="h-3.5 w-3.5" />
          </WorkbenchIconAction>
        )}
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 p-0">
        <Command>
          <CommandInput
            placeholder={page === "blocks" ? "Insert a block…" : "Embed an entity…"}
            value={search}
            onValueChange={setSearch}
          />
          <CommandList>
            {page === "blocks" ? (
              <>
                <CommandEmpty>No blocks found.</CommandEmpty>
                <CommandGroup heading="Basic blocks">
                  {SLASH_COMMANDS.map((command) => {
                    const Icon = BLOCK_ICON[command.id] ?? Slash;
                    return (
                      <CommandItem
                        key={command.id}
                        value={`${command.label} ${command.keywords.join(" ")}`}
                        onSelect={() => insertBlock(command.id)}
                      >
                        <Icon className="h-4 w-4 text-muted-foreground" />
                        <span className="flex-1">{command.label}</span>
                        <span className="text-micro text-muted-foreground">
                          {command.description}
                        </span>
                      </CommandItem>
                    );
                  })}
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Reference">
                  <CommandItem
                    value="embed entity link reference"
                    onSelect={() => {
                      setPage("embed");
                      setSearch("");
                    }}
                  >
                    <Link2 className="h-4 w-4 text-muted-foreground" />
                    <span className="flex-1">Embed entity…</span>
                  </CommandItem>
                </CommandGroup>
              </>
            ) : (
              <>
                <CommandItem
                  value="back"
                  onSelect={() => {
                    setPage("blocks");
                    setSearch("");
                  }}
                >
                  <ChevronLeft className="h-4 w-4 text-muted-foreground" />
                  <span>Back to blocks</span>
                </CommandItem>
                {error && <p className="px-3 py-2 text-label text-destructive">{error}</p>}
                {embedding && (
                  <p className="flex items-center gap-2 px-3 py-2 text-label text-muted-foreground">
                    <Loader2 className="mol-motion-progress-spin h-3.5 w-3.5" /> Embedding…
                  </p>
                )}
                <CommandEmpty>No entities to embed.</CommandEmpty>
                {embedGroups.map((group) => (
                  <CommandGroup key={group.kind} heading={group.label}>
                    {group.items.map((item) => (
                      <CommandItem
                        key={`${group.kind}:${item.id}`}
                        value={`${group.kind} ${item.label} ${item.id}`}
                        disabled={embedding}
                        onSelect={() => void embed(group.kind, item.id)}
                      >
                        <group.icon className={cn("h-4 w-4 text-muted-foreground")} />
                        <span className="flex-1 truncate">{item.label}</span>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};
