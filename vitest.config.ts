import { defineConfig } from "vitest/config";
import os from "node:os";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    // SQLite (better-sqlite3) holds file handles; the default threads pool
    // makes parallel workers contend for the same temp DBs on Windows
    // (EBUSY family: engine buildIndex cleanup, server ori_warmth,
    // edge-cases DB resilience). Forks isolate the native handles.
    pool: "forks",
    // MCP tests spawn a real server; embedding model load can be slow cold.
    testTimeout: 30000,
    hookTimeout: 120000,
    // Keep the update check and "what's new" state out of the developer's
    // real cache. Spawned MCP servers inherit this.
    env: { ORI_UPDATE_CACHE_DIR: path.join(os.tmpdir(), "ori-test-update-cache") },
  },
});
