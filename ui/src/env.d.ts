/// <reference types="@rsbuild/core/types" />

/**
 * Imports the SVG file as a React component.
 * @requires [@rsbuild/plugin-svgr](https://npmjs.com/package/@rsbuild/plugin-svgr)
 */
declare module "*.svg?react" {
  import type React from "react";

  const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
  export default ReactComponent;
}
declare const __USE_MOCK__: boolean;

declare module "plotly.js-cartesian-dist-min" {
  const Plotly: unknown;
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";

  type PlotData = Record<string, unknown>;
  type PlotLayout = Record<string, unknown>;
  type PlotConfig = Record<string, unknown>;

  export interface PlotComponentProps {
    data: PlotData[];
    layout?: Partial<PlotLayout>;
    config?: Partial<PlotConfig>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: unknown, graphDiv: HTMLDivElement) => void;
    onUpdate?: (figure: unknown, graphDiv: HTMLDivElement) => void;
    onRelayout?: (event: unknown) => void;
    onHover?: (event: unknown) => void;
  }

  export default function createPlotlyComponent(plotly: unknown): ComponentType<PlotComponentProps>;
}
