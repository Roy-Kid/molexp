/**
 * CommandPalette — `/`-triggered slash command autocomplete for chat inputs.
 *
 * Pairs a small ``useCommandPalette`` hook (state + filtering) with a
 * presentational component (popover) so any chat box can opt in by
 * routing its keydown/change handlers through the hook. The hook is
 * keyboard-only: it never owns the underlying textarea so callers retain
 * full control of submission / IME / autosizing.
 */

import { Layers, Sparkles } from "lucide-react";
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { type ApiCommand, commandsApi } from "@/app/state/api";
import { Badge } from "@/components/ui/badge";

const PALETTE_HEIGHT_PX = 220;

/**
 * Match a leading slash command at the start of the textarea content. We
 * intentionally accept partial input — ``/pl`` opens the palette filtered
 * to commands starting with ``pl``.
 */
const SLASH_PREFIX_RE = /^\s*\/([a-z0-9-]*)$/i;

/**
 * Parse the textarea value into a ``{ open, query }`` decision for the
 * palette. Pure / dependency-free so tests can exercise the grammar
 * without a React tree.
 */
export const parseSlashQuery = (value: string): { open: boolean; query: string } => {
  const match = SLASH_PREFIX_RE.exec(value);
  if (!match) return { open: false, query: "" };
  return { open: true, query: match[1] ?? "" };
};

export interface CommandPaletteState {
  open: boolean;
  query: string;
  filtered: ApiCommand[];
  activeIndex: number;
  /** Pure derivation — call from a textarea ``onChange``. */
  syncFromValue: (value: string) => void;
  /**
   * Forward a textarea ``onKeyDown`` event. Returns ``true`` when the
   * palette consumed the key — caller should ``preventDefault`` and skip
   * its own submission logic for that event.
   */
  handleKeyDown: (event: ReactKeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => boolean;
  /** Apply the active suggestion to ``value`` and return the new value. */
  applyActive: (value: string) => string | null;
  setActiveIndex: (idx: number) => void;
  close: () => void;
}

interface UseCommandPaletteOptions {
  /** Pre-loaded command list. When omitted, the hook fetches once on mount. */
  commands?: ApiCommand[];
}

/**
 * Hook that manages palette state for a chat input. The host textarea
 * keeps full control of the value; the hook only observes key/text
 * events and exposes the suggestion list + apply helper.
 */
export const useCommandPalette = (options: UseCommandPaletteOptions = {}): CommandPaletteState => {
  const [commands, setCommands] = useState<ApiCommand[]>(options.commands ?? []);
  const [query, setQuery] = useState<string>("");
  const [open, setOpen] = useState<boolean>(false);
  const [activeIndex, setActiveIndex] = useState<number>(0);

  // Lazy-fetch once when used without a pre-loaded list.
  useEffect(() => {
    if (options.commands !== undefined) return;
    let cancelled = false;
    commandsApi
      .list()
      .then((rows) => {
        if (!cancelled) setCommands(rows);
      })
      .catch(() => {
        // Soft fail — palette stays empty; chat still works.
        if (!cancelled) setCommands([]);
      });
    return () => {
      cancelled = true;
    };
  }, [options.commands]);

  const filtered = useMemo<ApiCommand[]>(() => {
    if (!open) return [];
    const needle = query.toLowerCase();
    if (!needle) return commands;
    return commands.filter(
      (c) => c.slashName.toLowerCase().startsWith(needle) || c.name.toLowerCase().includes(needle),
    );
  }, [commands, query, open]);

  // Keep activeIndex in range whenever the visible set shrinks/grows.
  useEffect(() => {
    if (filtered.length === 0) {
      setActiveIndex(0);
      return;
    }
    if (activeIndex >= filtered.length) setActiveIndex(0);
  }, [filtered.length, activeIndex]);

  const syncFromValue = useCallback((value: string) => {
    const next = parseSlashQuery(value);
    setOpen(next.open);
    setQuery(next.query);
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    setActiveIndex(0);
  }, []);

  const applyActive = useCallback(
    (value: string): string | null => {
      if (!open || filtered.length === 0) return null;
      const cmd = filtered[Math.min(activeIndex, filtered.length - 1)];
      // Replace only the leading slash token; preserve any trailing args
      // the user already typed (rare, but supports ``/pl<TAB>x=1`` style).
      const replaced = value.replace(SLASH_PREFIX_RE, (_, _slug, _offset) => {
        const head = `/${cmd.slashName}`;
        const trailingSpace = cmd.parameters.length > 0 ? " " : " ";
        return `${head}${trailingSpace}`;
      });
      close();
      return replaced;
    },
    [open, filtered, activeIndex, close],
  );

  const handleKeyDown = useCallback<CommandPaletteState["handleKeyDown"]>(
    (event) => {
      if (!open || filtered.length === 0) return false;
      switch (event.key) {
        case "ArrowUp":
          // Visual list grows upward (active=0 sits at the bottom near the
          // textarea), so ArrowUp moves to the next item further up the list.
          setActiveIndex((idx) => (idx + 1) % filtered.length);
          return true;
        case "ArrowDown":
          setActiveIndex((idx) => (idx - 1 + filtered.length) % filtered.length);
          return true;
        case "Tab":
          // Tab applies the suggestion; caller is expected to update the
          // textarea value via ``applyActive`` in its own handler.
          return true;
        case "Escape":
          close();
          return true;
        default:
          return false;
      }
    },
    [open, filtered.length, close],
  );

  return {
    open,
    query,
    filtered,
    activeIndex,
    syncFromValue,
    handleKeyDown,
    applyActive,
    setActiveIndex,
    close,
  };
};

interface CommandPaletteProps {
  state: CommandPaletteState;
  /** Anchor element — palette is positioned above it. */
  anchorRef: RefObject<HTMLElement | null>;
  onPick: (command: ApiCommand) => void;
}

/**
 * Presentational popover. Hidden when the hook reports ``open=false``
 * or has no filtered results.
 */
export const CommandPalette = ({
  state,
  anchorRef,
  onPick,
}: CommandPaletteProps): JSX.Element | null => {
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [position, setPosition] = useState<CSSProperties>({});

  // Anchor by the BOTTOM edge so the popover bottom stays glued just above
  // the textarea while the list grows upward as more results filter in.
  // Anchoring by ``top`` made the popover jitter as its rendered height
  // changed mid-typing ("框乱飞").
  useEffect(() => {
    if (!state.open || state.filtered.length === 0) return;
    const anchor = anchorRef.current;
    if (!anchor) return;
    const update = () => {
      const rect = anchor.getBoundingClientRect();
      setPosition({
        position: "fixed",
        left: rect.left,
        bottom: Math.max(8, window.innerHeight - rect.top + 6),
        width: rect.width,
        maxHeight: PALETTE_HEIGHT_PX,
      });
    };
    update();
    const resizeObserver = new ResizeObserver(update);
    resizeObserver.observe(anchor);
    window.addEventListener("scroll", update, true);
    window.addEventListener("resize", update);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("resize", update);
    };
  }, [state.open, state.filtered.length, anchorRef]);

  if (!state.open || state.filtered.length === 0) return null;

  return (
    <div
      ref={popoverRef}
      role="listbox"
      aria-label="Slash commands"
      className="z-50 overflow-y-auto rounded-md border border-border bg-popover shadow-md"
      style={position}
    >
      <ul className="flex flex-col-reverse divide-y divide-border divide-y-reverse text-sm">
        {state.filtered.map((cmd, idx) => {
          const active = idx === state.activeIndex;
          const Icon = cmd.isBuiltin ? Sparkles : Layers;
          return (
            <li key={`${cmd.isBuiltin ? "builtin" : "skill"}-${cmd.slashName}`}>
              <button
                type="button"
                role="option"
                aria-selected={active}
                onMouseDown={(e) => {
                  // Keep the textarea focused; mouseDown skips the blur
                  // that would otherwise close the palette before click.
                  e.preventDefault();
                }}
                onClick={() => onPick(cmd)}
                onMouseEnter={() => state.setActiveIndex(idx)}
                className={`flex w-full items-start gap-2 px-3 py-2 text-left transition ${
                  active ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
                }`}
              >
                <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 opacity-70" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs">/{cmd.slashName}</span>
                    <span className="truncate text-xs text-muted-foreground">{cmd.name}</span>
                    {cmd.defaultPlanMode ? (
                      <Badge variant="outline" className="text-[10px]">
                        plan
                      </Badge>
                    ) : null}
                    {cmd.isBuiltin ? (
                      <Badge variant="secondary" className="text-[10px]">
                        builtin
                      </Badge>
                    ) : null}
                  </div>
                  {cmd.description ? (
                    <div className="truncate text-xs text-muted-foreground">{cmd.description}</div>
                  ) : null}
                  {cmd.parameters.length > 0 ? (
                    <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                      {cmd.parameters.map((p) => `${p.name}=…`).join("  ")}
                    </div>
                  ) : null}
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
};
