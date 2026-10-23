import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    globals: false,
    testTimeout: 10000,
  },
  resolve: {
    alias: {
      "@main": resolve(__dirname, "src/main"),
    },
  },
});
