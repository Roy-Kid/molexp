import { defineConfig } from "@rstest/core";
import { withRsbuildConfig } from "@rstest/adapter-rsbuild";

export default defineConfig({
  extends: withRsbuildConfig(),
  browser: {
    enabled: false,
  },
  globals: true,
  testEnvironment: "node",
});
