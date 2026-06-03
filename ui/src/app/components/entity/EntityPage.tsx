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
    <span className="flex items-baseline gap-1 text-xs">
      <span className="font-semibold tabular-nums text-foreground">{value}</span>
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
};

interface EntityHeaderProps {
  icon: ComponentType<{ className?: string }>;
  title: string;
  subtitle?: string;
  status?: string;
  /** Inline element rendered after the status badge (e.g. "Live" indicator). */
  titleAccessory?: ReactNode;
  actions?: ReactNode;
  metrics?: ReactNode;
}

export const EntityHeader = ({
  icon: Icon,
  title,
  subtitle,
  status,
  titleAccessory,
  actions,
  metrics,
}: EntityHeaderProps): JSX.Element => {
  return (
    <section className="border-b border-border/70 bg-background">
      <div className="px-4 pt-2">
        {/* min-h locks the header height regardless of whether actions/metrics
            slots are populated, so different viewers (some with buttons, some
            without) line up vertically in the same way. */}
        <div className="mt-1.5 flex min-h-9 items-center justify-between gap-4 pb-2.5">
          <div className="flex min-w-0 flex-1 items-center gap-2.5">
            <div className="flex h-7 w-7 flex-none items-center justify-center rounded-md bg-muted">
              <Icon className="h-4 w-4 text-foreground" />
            </div>
            <div className="flex min-w-0 flex-1 items-center gap-2">
              <h2 className="truncate text-base font-semibold text-foreground">{title}</h2>
              {status && <StatusBadge status={status} />}
              {titleAccessory}
              {subtitle && (
                <span
                  className="hidden min-w-0 truncate text-xs text-muted-foreground md:inline"
                  title={subtitle}
                >
                  · {subtitle}
                </span>
              )}
            </div>
          </div>

          {(actions || metrics) && (
            <div className="flex flex-none items-center gap-4">
              {metrics && (
                <div className="flex flex-wrap items-baseline justify-end gap-x-3 gap-y-0.5">
                  {metrics}
                </div>
              )}
              {actions && <div className="flex items-center gap-1">{actions}</div>}
            </div>
          )}
        </div>
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
    <dl className="grid gap-x-6 gap-y-2 md:grid-cols-2">
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 flex-col">
          <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            {item.label}
          </dt>
          <dd className="mt-0.5 min-w-0 truncate text-sm text-foreground">{item.value}</dd>
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
  subtitle?: string;
  status?: string;
  actions?: ReactNode;
  metrics?: ReactNode;

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
  subtitle,
  status,
  actions,
  metrics,
  tabs,
  activeTab,
  defaultTab,
  onActiveTabChange,
  children,
}: EntityPageProps): JSX.Element => {
  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        icon={icon}
        title={title}
        subtitle={subtitle}
        status={status}
        actions={actions}
        metrics={metrics}
      />

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
