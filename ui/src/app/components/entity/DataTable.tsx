import type { ComponentType, JSX, ReactNode } from "react";
import { Fragment } from "react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";

export interface DataTableColumn<T> {
  key: string;
  header: ReactNode;
  width?: string;
  align?: "left" | "right";
  cell: (row: T) => ReactNode;
}

export interface DataTableRowAction<T> {
  id: string;
  label: string;
  icon?: ComponentType<{ className?: string }>;
  disabled?: boolean;
  destructive?: boolean;
  separatorBefore?: boolean;
  title?: string;
  onSelect: (row: T) => void;
}

export interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  data: T[];
  getRowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  empty: ReactNode;
  rowClassName?: (row: T) => string;
  rowActions?: (row: T) => DataTableRowAction<T>[];
}

export const DataTable = <T,>({
  columns,
  data,
  getRowKey,
  onRowClick,
  empty,
  rowClassName,
  rowActions,
}: DataTableProps<T>): JSX.Element => {
  const renderRow = (row: T): JSX.Element => {
    const actions = rowActions?.(row) ?? [];
    const rowElement = (
      <tr
        className={`group transition-colors hover:bg-accent/50 ${onRowClick ? "cursor-pointer" : ""} ${rowClassName?.(row) ?? ""}`}
        onClick={onRowClick ? () => onRowClick(row) : undefined}
      >
        {columns.map((col) => (
          <td key={col.key} className={`px-3 py-1.5 ${col.align === "right" ? "text-right" : ""}`}>
            {col.cell(row)}
          </td>
        ))}
      </tr>
    );

    if (actions.length === 0) {
      return <Fragment key={getRowKey(row)}>{rowElement}</Fragment>;
    }

    return (
      <ContextMenu key={getRowKey(row)}>
        <ContextMenuTrigger asChild>{rowElement}</ContextMenuTrigger>
        <ContextMenuContent>
          {actions.map((action) => {
            const ActionIcon = action.icon;
            return (
              <Fragment key={action.id}>
                {action.separatorBefore && <ContextMenuSeparator />}
                <ContextMenuItem
                  disabled={action.disabled}
                  title={action.title}
                  className={
                    action.destructive ? "text-destructive focus:text-destructive" : undefined
                  }
                  onSelect={() => {
                    if (!action.disabled) {
                      action.onSelect(row);
                    }
                  }}
                >
                  {ActionIcon && <ActionIcon className="mr-2 h-3.5 w-3.5" />}
                  <span className="truncate">{action.label}</span>
                </ContextMenuItem>
              </Fragment>
            );
          })}
        </ContextMenuContent>
      </ContextMenu>
    );
  };

  return (
    <div className="flex-1 overflow-auto">
      <table className="w-full text-left text-sm">
        <thead className="sticky top-0 z-10 border-b border-border bg-muted/40 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-3 py-1.5 ${col.width ?? "w-auto"} ${col.align === "right" ? "text-right" : ""}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="py-10">
                {empty}
              </td>
            </tr>
          ) : (
            data.map(renderRow)
          )}
        </tbody>
      </table>
    </div>
  );
};
