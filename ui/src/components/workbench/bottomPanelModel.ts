export type BottomPanelTab = "logs" | "problems" | "runs" | "artifacts";

export const BOTTOM_PANEL_TABS: readonly { id: BottomPanelTab; label: string }[] = [
  { id: "logs", label: "Logs" },
  { id: "problems", label: "Problems" },
  { id: "runs", label: "Runs" },
  { id: "artifacts", label: "Artifacts" },
];

export const BOTTOM_PANEL_DEFAULT_HEIGHT = 220;
export const BOTTOM_PANEL_MIN_HEIGHT = 120;
export const BOTTOM_PANEL_KEYBOARD_STEP = 16;

const STORAGE_KEY = "molexp.workbench.bottomPanel";
const MAX_HEIGHT_RATIO = 0.5;

export interface PersistedBottomPanel {
  open: boolean;
  tab: BottomPanelTab;
  height: number;
}

const defaultState = (): PersistedBottomPanel => ({
  open: false,
  tab: "logs",
  height: BOTTOM_PANEL_DEFAULT_HEIGHT,
});

export const maxBottomPanelHeight = (): number =>
  typeof window === "undefined"
    ? BOTTOM_PANEL_DEFAULT_HEIGHT * 2
    : Math.max(BOTTOM_PANEL_MIN_HEIGHT, Math.floor(window.innerHeight * MAX_HEIGHT_RATIO));

export const readBottomPanelState = (): PersistedBottomPanel => {
  if (typeof window === "undefined") return defaultState();

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState();

    const parsed = JSON.parse(raw) as Partial<PersistedBottomPanel>;
    const tab = BOTTOM_PANEL_TABS.some((item) => item.id === parsed.tab)
      ? (parsed.tab as BottomPanelTab)
      : "logs";
    const height =
      typeof parsed.height === "number" && parsed.height >= BOTTOM_PANEL_MIN_HEIGHT
        ? parsed.height
        : BOTTOM_PANEL_DEFAULT_HEIGHT;

    return { open: Boolean(parsed.open), tab, height };
  } catch {
    return defaultState();
  }
};

export const persistBottomPanelState = (state: PersistedBottomPanel): void => {
  if (typeof window === "undefined") return;

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Local storage may be unavailable or full; panel state remains usable.
  }
};
