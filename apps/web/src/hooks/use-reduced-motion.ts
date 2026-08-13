import { useEffect, useState } from "react";

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

const getMotionQuery = (): MediaQueryList | null =>
  typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(REDUCED_MOTION_QUERY)
    : null;

/** Reactively track the user's non-essential motion preference. */
export const useReducedMotion = (): boolean => {
  const [reduced, setReduced] = useState(() => getMotionQuery()?.matches ?? false);

  useEffect(() => {
    const query = getMotionQuery();
    if (!query) return;

    const onChange = (event: MediaQueryListEvent): void => setReduced(event.matches);

    setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
};
