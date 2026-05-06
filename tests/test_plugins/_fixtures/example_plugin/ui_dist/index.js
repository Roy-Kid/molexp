// Example third-party plugin ESM bundle (test fixture).
//
// A real third-party package would ship its built bundle here and the
// frontend would dynamic-import it at runtime. We only need a tiny
// shape-correct stub so the runtime test can assert the file is served
// and the loader calls `register()`.

const RENDERER_MARKER = "molexp-example-plugin-renderer";

const examplePlugin = {
  id: "example",
  marker: RENDERER_MARKER,
  register() {
    if (typeof globalThis !== "undefined") {
      globalThis.__molexpExamplePluginRegistered = true;
      globalThis.__molexpExamplePluginMarker = RENDERER_MARKER;
    }
  },
};

export default examplePlugin;
