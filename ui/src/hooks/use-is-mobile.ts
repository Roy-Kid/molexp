import { useEffect, useState } from "react";

/** Viewport width (px) at and above which the desktop 3-pane shell is used. */
export const MOBILE_BREAKPOINT = 768;

/**
 * Track whether the viewport is narrower than `breakpoint` (default 768px / Tailwind `md`).
 *
 * Used by the shell to swap its fixed 3-column resizable layout for a single-column
 * layout with the nav + inspector moved into edge drawers on small screens. SPA-only
 * (Vite), but the `window` guard keeps the initial value safe under any non-DOM render.
 */
export function useIsMobile(breakpoint: number = MOBILE_BREAKPOINT): boolean {
  const [isMobile, setIsMobile] = useState<boolean>(() =>
    typeof window === "undefined" ? false : window.innerWidth < breakpoint,
  );

  useEffect(() => {
    const query = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
    const onChange = (): void => setIsMobile(window.innerWidth < breakpoint);
    query.addEventListener("change", onChange);
    onChange();
    return () => query.removeEventListener("change", onChange);
  }, [breakpoint]);

  return isMobile;
}
