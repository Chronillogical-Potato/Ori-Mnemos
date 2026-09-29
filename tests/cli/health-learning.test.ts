/**
 * `ori health` must say out loud when learning is not happening.
 *
 * `getLearningHealth` has reported `neverUpdated` since fix-list item 5, but
 * a number inside a JSON blob is not a warning. The production table this was
 * written against: 807 tracked notes, 713 never updated, all 713 of them shown
 * to an agent - retrieval running, session-end credit not. Nothing displayed
 * that until the check below existed.
 *
 * The threshold (majority of at least 50 tracked notes) is a boundary a
 * plausible edit would move, so both sides of it are pinned.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import { initDB } from "../../src/core/engine.js";
import { initQValueTables, incrementExposure, updateQ, logRetrieval } from "../../src/core/qvalue.js";
import { runHealth } from "../../src/cli/health.js";
import { runInit } from "../../src/cli/init.js";

let vault: string;

beforeEach(async () => {
  vault = await fs.mkdtemp(path.join(os.tmpdir(), "ori-health-"));
  await runInit({ targetDir: vault });
});

afterEach(async () => {
  await fs.rm(vault, { recursive: true, force: true });
});

function seedSessions(sessions: number, credited: number, cliQueries = 0): void {
  const db = initDB(path.join(vault, ".ori", "embeddings.db"));
  try {
    initQValueTables(db);
    for (let s = 0; s < sessions; s++) {
      logRetrieval(db, `mcp-${s}`, "q", "semantic", `note-${s}`, 0, 1, 0, 0, 1);
      incrementExposure(db, `note-${s}`);
      if (s < credited) updateQ(db, `note-${s}`, -0.15, `mcp-${s}`, "session_batch");
    }
    // CLI traffic: exposure and retrieval_log, never credit (#37).
    for (let i = 0; i < cliQueries; i++) {
      logRetrieval(db, `cli-${i}`, "q", "semantic", `cli-note-${i}`, 0, 1, 0, 0, 1);
      incrementExposure(db, `cli-note-${i}`);
    }
  } finally {
    db.close();
  }
}

async function learningWarning(): Promise<string | undefined> {
  const result = await runHealth(vault);
  return result.warnings.find((w) => w.includes("never received a Q-update"));
}

describe("ori health: no false 'never credited' alarm (#37)", () => {
  // The session/note ratio check was removed: exposure without credit is now
  // normal (CLI queries, neutral and dead-end outcomes). It must not come back.
  it("stays quiet on agent sessions that read without crediting", async () => {
    seedSessions(20, 0, 200);
    expect(await learningWarning()).toBeUndefined();
  });
});
