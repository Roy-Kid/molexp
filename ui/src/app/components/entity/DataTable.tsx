import type { JSX, ReactNode } from "react";

export interface DataTableColumn<T> {
  key: string;
  header: ReactNode;
  width?: string;
  align?: "left" | "right";
  cell: (row: T) => ReactNode;
}

export interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  data: T[];
  getRowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  empty: ReactNode;
  rowClassName?: (row: T) => string;
}

export const DataTable = <T,>({
  columns,
  data,
  getRowKey,
  onRowClick,
  empty,
  rowClassName,
}: DataTableProps<T>): JSX.Element => {
  return (
    <div className="flex-1 overflow-auto">
      <table className="w-full text-sm text-left">
        <thead className="sticky top-0 border-b border-border/70 bg-background text-muted-foreground">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`py-3 px-6 ${col.width ?? "w-auto"} ${col.align === "right" ? "text-right" : ""}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="py-12">
                {empty}
              </td>
            </tr>
          ) : (
            data.map((row) => (
              <tr
                key={getRowKey(row)}
                className={`group transition-colors hover:bg-muted/40 ${onRowClick ? "cursor-pointer" : ""} ${rowClassName?.(row) ?? ""}`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`py-3 px-6 ${col.align === "right" ? "text-right" : ""}`}
                  >
                    {col.cell(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};
