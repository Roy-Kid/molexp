import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Docs: https://rsbuild.rs/config/
export default defineConfig(({ command }) => {
  const useMock = command === 'dev' && process.env.MOLEXP_USE_MOCK === 'true';

  return {
    plugins: [pluginReact()],
    resolve: {
      alias: {
        '@schemas': path.resolve(__dirname, '../src/molexp/schemas'),
      },
    },
    tools: {
      rspack: {
        resolve: {
          // Workaround for the published @molcrafts/molvis-core@0.0.7 tarball:
          // its trajectory worker is spawned via
          // `new Worker(new URL("./worker.ts", import.meta.url))`, but rslib
          // builds the package bundleless and leaves that URL string verbatim,
          // so the file actually shipped is worker.js. rsbuild 2 / rspack
          // resolves the worker entry with a dedicated resolver that ignores
          // `extensionAlias`, so map the exact published `.ts` path to the real
          // `.js` file. Fixed at source in molvis-core (now references
          // `./worker.js`); remove this once molexp depends on a release that
          // includes that fix.
          alias: {
            [path.resolve(
              __dirname,
              '../node_modules/@molcrafts/molvis-core/dist/transport/trajectory_worker/worker.ts',
            )]: path.resolve(
              __dirname,
              '../node_modules/@molcrafts/molvis-core/dist/transport/trajectory_worker/worker.js',
            ),
          },
        },
      },
    },
    source: {
      define: {
        __USE_MOCK__: useMock,
      },
    },
    server: {
      proxy: {
        '/api': {
          target: `http://localhost:${(() => {
            const arg = process.argv.find(a => a.startsWith('--api-port='));
            return arg ? arg.split('=')[1] : '8000';
          })()}`,
          changeOrigin: true,
        },
      },
    },
  };
});
