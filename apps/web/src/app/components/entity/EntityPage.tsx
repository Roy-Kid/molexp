import type { ComponentType, JSX, ReactNode } from "react";
import type { SemanticStatus } from "@/app/types";
import { EntityTabBar, EntityTabContent, EntityTabs } from "./EntityTabs";
import { StatusBadge } from "./StatusBadge";

interface SemanticStatusBadgeProps {
  status: SemanticStatus;
}

export const SemanticStatusBadge = ({ status }: SemanticStatusBadgeProps): JSX.Element => {
  return <StatusBadge status={status} />;
};

interface EntityMetricProps {
  label: string;
  value: number | string;
}

export const EntityMetric = ({ label, value }: EntityMetricProps): JSX.Element => {
  return (
    <span className="flex items-baseline gap-1 text-label">
      <span className="font-semibold tabular-nums text-foreground">{value}</span>
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
};

interface EntityHeaderProps {
  icon: ComponentType<{ className?: string }>;
  title: string;
  /** Hover tooltip on the title, e.g. the untruncated source text. */
  titleTooltip?: string;
  /** Primary / secondary controls — prefer one primary icon + a More menu. */
  actions?: ReactNode;
}

export const EntityHeader = ({
  icon: Icon,
  title,
  titleTooltip,
  actions,
}: EntityHeaderProps): JSX.Element => {
  return (
    <section className="bg-surface">
      <div className="flex h-toolbar min-w-0 items-center gap-2 px-2">
        <div className="hidden size-7 flex-none items-center justify-center text-accent sm:flex">
          <Icon className="size-4" aria-hidden />
        </div>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h2
            className="truncate text-title font-semibold tracking-tight text-foreground"
            title={titleTooltip ?? title}
          >
            {title}
          </h2>
        </div>
        {actions && <div className="flex flex-none items-center gap-0.5">{actions}</div>}
      </div>
    </section>
  );
};

interface KeyValueItem {
  label: string;
  value: ReactNode;
}

interface KeyValueGridProps {
  items: KeyValueItem[];
}

export const KeyValueGrid = ({ items }: KeyValueGridProps): JSX.Element => {
  return (
    <dl className="grid gap-x-6 gap-y-3 md:grid-cols-2">
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 flex-col">
          <dt className="text-label text-muted-foreground">{item.label}</dt>
          <dd className="mt-1 min-w-0 truncate text-body-lg text-foreground">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
};

// ── EntityPage ──────────────────────────────────────────────────────────────
//
// One template every entity viewer renders into. Owns the outer flex column,
// the header chrome (via :class:`EntityHeader`), and the tab bar + tab content
// pattern. Viewers only supply the data: header props, tab list, optional
// post-tab body. This is what keeps Workflow / Run / Asset / Project / Agent
// settings looking identical regardless of which tab is active.

export interface EntityPageTab {
  value: string;
  label: ReactNode;
  /** Tab body. Rendered inside an ``EntityTabContent`` so it owns scrolling. */
  content: ReactNode;
  /** Disable the tab trigger; used for plugin-discovered tabs that haven't loaded. */
  disabled?: boolean;
}

interface EntityPageProps {
  // Header — forwarded verbatim to :class:`EntityHeader`.
  icon: ComponentType<{ className?: string }>;
  title: string;
  titleTooltip?: string;
  actions?: ReactNode;

  // Tabs — controlled (pass ``activeTab`` + ``onActiveTabChange``) or
  // uncontrolled (pass ``defaultTab``). Omit ``tabs`` entirely for a
  // header-only page (``children`` then renders directly under the header).
  tabs?: EntityPageTab[];
  activeTab?: string;
  defaultTab?: string;
  onActiveTabChange?: (value: string) => void;

  /** Body rendered when ``tabs`` is omitted. */
  children?: ReactNode;
}

export const EntityPage = ({
  icon,
  title,
  titleTooltip,
  actions,
  tabs,
  activeTab,
  defaultTab,
  onActiveTabChange,
  children,
}: EntityPageProps): JSX.Element => {
  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader icon={icon} title={title} titleTooltip={titleTooltip} actions={actions} />

      {tabs && tabs.length > 0 ? (
        <EntityTabs
          value={activeTab}
          defaultValue={defaultTab ?? tabs[0]?.value}
          onValueChange={onActiveTabChange}
        >
          <EntityTabBar
            tabs={tabs.map(({ value, label, disabled }) => ({ value, label, disabled }))}
          />
          {tabs.map((tab) => (
            <EntityTabContent key={tab.value} value={tab.value}>
              {tab.content}
            </EntityTabContent>
          ))}
        </EntityTabs>
      ) : (
        children && <div className="flex flex-1 flex-col overflow-hidden">{children}</div>
      )}
    </div>
  );
};
