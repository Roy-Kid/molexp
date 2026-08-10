/**
 * Settings shell: left category nav + scrollable sections.
 *
 * Domain-free block — shared via molcrafts-ui registry.
 */

import type { JSX, ReactNode } from "react";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

export interface SettingsNavEntry {
  id: string;
  label: string;
  icon?: ReactNode;
  group?: string;
  groupLabel?: string;
  /** Section body (already wrapped in SettingsSection or free form). */
  content: ReactNode;
  /** Hide from nav + content (e.g. admin-only). */
  hidden?: boolean;
}

function NavSeparator({ label }: { label: string }): JSX.Element {
  return (
    <div className="flex items-center gap-2 px-1.5 pb-1 pt-2.5">
      <Separator className="min-w-0 flex-1" />
      <span className="shrink-0 text-[0.65rem] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <Separator className="min-w-0 flex-1" />
    </div>
  );
}

interface SettingsShellProps {
  title?: string;
  entries: SettingsNavEntry[];
  defaultId?: string;
  className?: string;
}

export function SettingsShell({
  title = "Settings",
  entries,
  defaultId,
  className,
}: SettingsShellProps): JSX.Element {
  const visible = useMemo(() => entries.filter((e) => !e.hidden), [entries]);
  const [activeId, setActiveId] = useState(defaultId ?? visible[0]?.id ?? "");
  const scrollRef = useRef<HTMLDivElement>(null);
  const navLockUntil = useRef(0);
  const categoryIds = useMemo(() => visible.map((s) => s.id), [visible]);

  useEffect(() => {
    if (!visible.some((e) => e.id === activeId) && visible[0]) {
      setActiveId(visible[0].id);
    }
  }, [visible, activeId]);

  const scrollToCategory = useCallback((id: string) => {
    const root = scrollRef.current;
    if (!root) return;
    const target = root.querySelector<HTMLElement>(`[data-settings-section="${id}"]`);
    if (!target) return;

    setActiveId(id);
    navLockUntil.current = performance.now() + 450;

    const top =
      target.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop;
    root.scrollTo({ top: Math.max(0, top - 8), behavior: "smooth" });
  }, []);

  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;

    const nodes = categoryIds
      .map((id) => root.querySelector<HTMLElement>(`[data-settings-section="${id}"]`))
      .filter((el): el is HTMLElement => el != null);

    if (nodes.length === 0) return;

    const updateActive = (): void => {
      if (performance.now() < navLockUntil.current) return;

      const rootTop = root.getBoundingClientRect().top;
      const threshold = rootTop + root.clientHeight * 0.28;
      let current = nodes[0].dataset.settingsSection ?? nodes[0].id;

      for (const node of nodes) {
        const top = node.getBoundingClientRect().top;
        if (top <= threshold) {
          current = node.dataset.settingsSection ?? node.id;
        } else {
          break;
        }
      }
      setActiveId((prev) => (prev === current ? prev : current));
    };

    updateActive();
    root.addEventListener("scroll", updateActive, { passive: true });
    return () => root.removeEventListener("scroll", updateActive);
  }, [categoryIds]);

  return (
    <div
      className={cn(
        "flex h-full min-h-0 w-full flex-col overflow-hidden rounded-control border border-border bg-background",
        className,
      )}
    >
      <header className="shrink-0 border-b border-border/70 px-5 py-3.5">
        <h2 className="text-title font-semibold tracking-tight">{title}</h2>
      </header>

      <div className="flex min-h-0 flex-1">
        <nav
          aria-label="Settings categories"
          className="flex w-[9.5rem] shrink-0 flex-col gap-0.5 overflow-y-auto border-r border-border/70 bg-muted/20 p-2 sm:w-44"
        >
          {visible.map((item, index) => {
            const isActive = activeId === item.id;
            const prev = visible[index - 1];
            const showGroup =
              item.group &&
              item.group !== "general" &&
              prev?.group !== item.group;
            return (
              <Fragment key={item.id}>
                {showGroup ? (
                  <NavSeparator label={item.groupLabel ?? item.group ?? ""} />
                ) : null}
                <button
                  type="button"
                  onClick={() => scrollToCategory(item.id)}
                  aria-current={isActive ? "true" : undefined}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-control border-l-2 px-2.5 py-1.5 text-left text-micro transition-colors duration-(--motion-fast) ease-standard",
                    isActive
                      ? "border-accent bg-accent/12 font-medium text-foreground"
                      : "border-transparent text-muted-foreground hover:bg-interactive hover:text-foreground",
                  )}
                >
                  {item.icon ? (
                    <span
                      className={cn(
                        "shrink-0",
                        isActive ? "text-accent" : "text-muted-foreground",
                      )}
                    >
                      {item.icon}
                    </span>
                  ) : null}
                  <span className="truncate">{item.label}</span>
                </button>
              </Fragment>
            );
          })}
        </nav>

        <div
          ref={scrollRef}
          className="min-w-0 flex-1 overflow-y-auto overscroll-contain"
        >
          <div className="space-y-0 px-5 py-4 sm:px-6">
            {visible.map((section, index) => (
              <div
                key={section.id}
                className={cn(
                  "py-5 first:pt-1 last:pb-8",
                  index < visible.length - 1 && "border-b border-border/60",
                )}
              >
                {section.content}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
