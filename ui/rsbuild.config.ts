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
    source: {
      define: {
        __USE_MOCK__: useMock,
      },
    },
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
