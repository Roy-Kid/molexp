import type * as React from "react";

declare global {
  namespace JSX {
    type Element = React.JSX.Element;
    interface IntrinsicElements extends React.JSX.IntrinsicElements {}
    interface IntrinsicAttributes extends React.JSX.IntrinsicAttributes {}
    interface ElementChildrenAttribute extends React.JSX.ElementChildrenAttribute {}
    interface LibraryManagedAttributes<C, P> extends React.JSX.LibraryManagedAttributes<C, P> {}
  }
}
