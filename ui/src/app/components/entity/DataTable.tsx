import type { ComponentType, JSX, ReactNode } from "react";
import { Fragment } from "react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ROW_PADDING_DENSE } from "./density";

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
  onRowActivate?: (row: T) => void;
  getRowLabel?: (row: T) => string;
  empty: ReactNode;
  rowClassName?: (row: T) => string;
  rowActions?: (row: T) => DataTableRowAction<T>[];
}

export const DataTable = <T,>({
  columns,
  data,
  getRowKey,
  onRowActivate,
  getRowLabel,
  empty,
  rowClassName,
  rowActions,
}: DataTableProps<T>): JSX.Element => {
  const renderRow = (row: T): JSX.Element => {
    const actions = rowActions?.(row) ?? [];
    const interactive = Boolean(onRowActivate);
    const activate = (): void => onRowActivate?.(row);
    const rowElement = (
      <TableRow
        tabIndex={interactive ? 0 : undefined}
        aria-label={interactive ? (getRowLabel?.(row) ?? `Open ${getRowKey(row)}`) : undefined}
        className={`group transition-colors hover:bg-interactive/50 focus-visible:bg-interactive/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring ${interactive ? "cursor-pointer" : ""} ${rowClassName?.(row) ?? ""}`}
        onClick={interactive ? activate : undefined}
        onKeyDown={
          interactive
            ? (event) => {
                if (
                  event.target !== event.currentTarget ||
                  (event.key !== "Enter" && event.key !== " ")
                ) {
                  return;
                }
                event.preventDefault();
                activate();
              }
            : undefined
        }
      >
        {columns.map((col) => (
          <TableCell
            key={col.key}
            className={`${ROW_PADDING_DENSE} ${col.align === "right" ? "text-right" : ""}`}
          >
            {col.cell(row)}
          </TableCell>
        ))}
      </TableRow>
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
      <Table className="w-full text-left text-body">
        <TableHeader className="sticky top-0 z-10 border-b border-border bg-background text-micro font-medium uppercase tracking-wide text-muted-foreground">
          <TableRow>
            {columns.map((col) => (
              <TableHead
                key={col.key}
                className={`${ROW_PADDING_DENSE} ${col.width ?? "w-auto"} ${col.align === "right" ? "text-right" : ""}`}
              >
                {col.header}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody className="divide-y divide-border/50">
          {data.length === 0 ? (
            <TableRow>
              <TableCell colSpan={columns.length} className="py-8">
                {empty}
              </TableCell>
            </TableRow>
          ) : (
            data.map(renderRow)
          )}
        </TableBody>
      </Table>
    </div>
  );
};
