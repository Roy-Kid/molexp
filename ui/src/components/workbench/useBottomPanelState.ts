import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  BOTTOM_PANEL_DEFAULT_HEIGHT,
  BOTTOM_PANEL_KEYBOARD_STEP,
  BOTTOM_PANEL_MIN_HEIGHT,
  BOTTOM_PANEL_TABS,
  type BottomPanelTab,
  maxBottomPanelHeight,
  persistBottomPanelState,
  readBottomPanelState,
} from "./bottomPanelModel";

export const useBottomPanelState = () => {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<BottomPanelTab>("logs");
  const [height, setHeight] = useState(BOTTOM_PANEL_DEFAULT_HEIGHT);
  const [resizing, setResizing] = useState(false);
  const dragStart = useRef<{ y: number; height: number } | null>(null);

  useEffect(() => {
    const persisted = readBottomPanelState();
    setOpen(persisted.open);
    setTab(persisted.tab);
    setHeight(Math.min(maxBottomPanelHeight(), persisted.height));
  }, []);

  const selectTab = useCallback(
    (nextTab: BottomPanelTab) => {
      setTab(nextTab);
      setOpen(true);
      persistBottomPanelState({ open: true, tab: nextTab, height });
    },
    [height],
  );

  const toggleOpen = useCallback(() => {
    setOpen((current) => {
      const next = !current;
      persistBottomPanelState({ open: next, tab, height });
      return next;
    });
  }, [height, tab]);

  const onResizePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLHRElement>) => {
      event.preventDefault();
      dragStart.current = { y: event.clientY, height };
      setResizing(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [height],
  );

  const onResizePointerMove = useCallback((event: ReactPointerEvent<HTMLHRElement>) => {
    if (!dragStart.current) return;

    const delta = dragStart.current.y - event.clientY;
    const next = Math.min(
      maxBottomPanelHeight(),
      Math.max(BOTTOM_PANEL_MIN_HEIGHT, dragStart.current.height + delta),
    );
    setHeight(next);
  }, []);

  const onResizePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLHRElement>) => {
      if (!dragStart.current) return;

      dragStart.current = null;
      setResizing(false);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      setHeight((current) => {
        persistBottomPanelState({ open: true, tab, height: current });
        return current;
      });
    },
    [tab],
  );

  const onResizeKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLHRElement>) => {
      const maximumHeight = maxBottomPanelHeight();
      const step = event.shiftKey ? BOTTOM_PANEL_KEYBOARD_STEP * 2 : BOTTOM_PANEL_KEYBOARD_STEP;
      let resolveNext: ((current: number) => number) | undefined;

      switch (event.key) {
        case "ArrowUp":
          resolveNext = (current) => Math.min(maximumHeight, current + step);
          break;
        case "ArrowDown":
          resolveNext = (current) => Math.max(BOTTOM_PANEL_MIN_HEIGHT, current - step);
          break;
        case "Home":
          resolveNext = () => BOTTOM_PANEL_MIN_HEIGHT;
          break;
        case "End":
          resolveNext = () => maximumHeight;
          break;
        default:
          return;
      }

      event.preventDefault();
      setHeight((current) => {
        const next = resolveNext?.(current) ?? current;
        persistBottomPanelState({ open: true, tab, height: next });
        return next;
      });
    },
    [tab],
  );

  const onTabKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>, currentTab: BottomPanelTab) => {
      const index = BOTTOM_PANEL_TABS.findIndex((item) => item.id === currentTab);
      let nextIndex: number | undefined;

      if (event.key === "ArrowRight") {
        nextIndex = (index + 1) % BOTTOM_PANEL_TABS.length;
      }
      if (event.key === "ArrowLeft") {
        nextIndex = (index - 1 + BOTTOM_PANEL_TABS.length) % BOTTOM_PANEL_TABS.length;
      }
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = BOTTOM_PANEL_TABS.length - 1;
      if (nextIndex === undefined) return;

      event.preventDefault();
      const nextTab = BOTTOM_PANEL_TABS[nextIndex].id;
      selectTab(nextTab);
      window.requestAnimationFrame(() => {
        document.getElementById(`bottom-tab-${nextTab}`)?.focus();
      });
    },
    [selectTab],
  );

  return {
    height,
    maximumHeight: maxBottomPanelHeight(),
    onResizeKeyDown,
    onResizePointerDown,
    onResizePointerMove,
    onResizePointerUp,
    onTabKeyDown,
    open,
    resizing,
    selectTab,
    tab,
    toggleOpen,
  };
};
