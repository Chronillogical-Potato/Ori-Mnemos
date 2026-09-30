/**
 * #37 end to end over a real `ori serve --mcp`: the agent is told, the agent
 * relays, the tool applies the user's answer. The reset must never happen
 * without the tool call.
 */
import { describe, it, expect, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import { createMcpTestContext, callTool, type McpTestContext } from "./harness.js";
import { initQValueTables } from "../../src/core/qvalue.js";

let ctx: McpTestContext | null = null;

afterEach(async () => {
  await ctx?.cleanup();
  ctx = null;
});

async function seededContext(
  extra?: (db: InstanceType<typeof Database>) => void,
): Promise<{ ctx: McpTestContext; dbPath: string }> {
  // 1. Start once so init + scaffold exist, then stop.
  const first = await createMcpTestContext();
  const vault = first.vaultDir;
  await first.client.close();
  // 2. Seed an index with legacy learning.
  const dbPath = path.join(vault, ".ori", "embeddings.db");
  await fs.mkdir(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  db.exec("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)");
  initQValueTables(db);
  const ins = db.prepare(
    "INSERT INTO q_history (note_id, old_q, new_q, reward, reward_source, session_id) VALUES (?,?,?,?,?,?)",
  );
  for (let n = 0; n < 5; n++) {
    let q = 0.5;
    for (let u = 0; u < 6; u++) {
      ins.run(`note-${n}`, q, q * 0.9, 0, "session_batch", `cli-${n}-${u}`);
      q *= 0.9;
    }
    db.prepare("INSERT INTO note_q (note_id, q_value, update_count, exposure_count) VALUES (?,?,6,8)").run(`note-${n}`, q);
  }
  extra?.(db);
  db.close();
  // 3. Start a server on the seeded vault.
  const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
  const { StdioClientTransport } = await import("@modelcontextprotocol/sdk/client/stdio.js");
  const transport = new StdioClientTransport({
    command: "node",
    args: [path.resolve(import.meta.dirname, "..", "..", "dist", "index.js"), "serve", "--mcp"],
    cwd: vault,
    stderr: "pipe",
  });
  const client = new Client({ name: "ori-test-client", version: "1.0.0" }, { capabilities: {} });
  await client.connect(transport);
  const c: McpTestContext = {
    client,
    transport,
    vaultDir: vault,
    cleanup: async () => {
      try { await client.close(); } catch { /* closed */ }
      await fs.rm(vault, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    },
  };
  return { ctx: c, dbPath };
}

function count(dbPath: string, sql: string): number {
  const db = new Database(dbPath, { readonly: true });
  try {
    return (db.prepare(sql).get() as { c: number }).c;
  } finally {
    db.close();
  }
}

describe("#37 learning reset over MCP", () => {
  it("wake carries the notice once per session, and nothing is reset without the tool", async () => {
    const s = await seededContext();
    ctx = s.ctx;
    const first = (await callTool(ctx.client, "ori_wake")).parsed as Record<string, unknown>;
    expect(String(first.learning_reset_notice)).toContain("ori_learning_reset");
    const second = (await callTool(ctx.client, "ori_wake")).parsed as Record<string, unknown>;
    expect(second).not.toHaveProperty("learning_reset_notice");
    expect(count(s.dbPath, "SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(5);
  });

  it("the cap holds across server restarts", async () => {
    const s = await seededContext();
    ctx = s.ctx;
    const vault = ctx.vaultDir;
    const shown: boolean[] = [];
    for (let session = 0; session < 4; session++) {
      if (session > 0) {
        await ctx.client.close();
        const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
        const { StdioClientTransport } = await import("@modelcontextprotocol/sdk/client/stdio.js");
        const transport = new StdioClientTransport({
          command: "node",
          args: [path.resolve(import.meta.dirname, "..", "..", "dist", "index.js"), "serve", "--mcp"],
          cwd: vault,
          stderr: "pipe",
        });
        const client = new Client({ name: "ori-test-client", version: "1.0.0" }, { capabilities: {} });
        await client.connect(transport);
        ctx = { ...ctx, client, transport };
      }
      const wake = (await callTool(ctx.client, "ori_wake")).parsed as Record<string, unknown>;
      shown.push("learning_reset_notice" in wake);
    }
    expect(shown).toEqual([true, true, true, false]);
    expect(count(s.dbPath, "SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(5);
    // cleanup() closes the client it was built with; the live one is the last
    // restart's, and Windows will not remove a directory its server holds open.
    await ctx.client.close();
  });

  it("accepted via the tool: learned state cleared, backup written, not asked again", async () => {
    const s = await seededContext();
    ctx = s.ctx;
    await callTool(ctx.client, "ori_wake"); // the agent sees the notice first
    const r = (await callTool(ctx.client, "ori_learning_reset", { decision: "accepted" })).parsed as Record<string, unknown>;
    expect(r).toMatchObject({ success: true, decision: "accepted", notesCleared: 5, historyArchived: 30 });
    expect(typeof r.backup).toBe("string");
    await fs.access(r.backup as string);
    expect(count(s.dbPath, "SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(0);
    const wake = (await callTool(ctx.client, "ori_wake")).parsed as Record<string, unknown>;
    expect(wake).not.toHaveProperty("learning_reset_notice");
  });

  it("declined via the tool: learning kept", async () => {
    const s = await seededContext();
    ctx = s.ctx;
    const r = (await callTool(ctx.client, "ori_learning_reset", { decision: "declined" })).parsed as Record<string, unknown>;
    expect(r).toMatchObject({ success: true, decision: "declined", notesCleared: 0 });
    expect(count(s.dbPath, "SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(5);
  });

  it("an accept the agent was never prompted for is refused", async () => {
    const s = await seededContext();
    ctx = s.ctx;
    const r = (await callTool(ctx.client, "ori_learning_reset", { decision: "accepted" })).parsed as Record<string, unknown>;
    expect(r).toMatchObject({ success: true, skipped: "not offered", notesCleared: 0 });
    expect(count(s.dbPath, "SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(5);
  });

  it("a fresh vault is never asked", async () => {
    ctx = await createMcpTestContext();
    const wake = (await callTool(ctx.client, "ori_wake")).parsed as Record<string, unknown>;
    expect(wake).not.toHaveProperty("learning_reset_notice");
  });
});

// #37 follow-up (Codex review): a checkpoint left by a killed 0.7.0 session
// holds rewards computed under the defective rules, including dead-end
// penalties. Replayed on the first start after upgrading, it was written as
// fixed-rule learning (rule = 1), negative Q included.
describe("checkpoint recovery across the #37 rule change", () => {
  it("discards pre-fix checkpoints and applies stamped ones", async () => {
    const s = await seededContext((db) => {
      db.exec("CREATE TABLE IF NOT EXISTS session_checkpoint (session_id TEXT PRIMARY KEY, rewards_json TEXT NOT NULL, updated_at TEXT NOT NULL)");
      const put = db.prepare("INSERT INTO session_checkpoint VALUES (?, ?, datetime('now', ?))");
      put.run("old-session", JSON.stringify({ "old-dead-end": -0.15, "old-neutral": 0 }), "-1 hour");
      put.run("new-session", JSON.stringify({ rule: 37, rewards: { "cited-note": 1.0 } }), "-1 hour");
      // Touched seconds ago: may belong to a server still running on this
      // vault, which will flush it itself. Recovering it too credited twice.
      put.run("live-session", JSON.stringify({ rule: 37, rewards: { "live-note": 1.0 } }), "-10 seconds");
    });
    ctx = s.ctx;
    await callTool(ctx.client, "ori_wake"); // recovery runs at server start
    const db = new Database(s.dbPath, { readonly: true });
    try {
      const rows = db.prepare("SELECT note_id, update_count, rule FROM note_q WHERE note_id IN ('old-dead-end', 'old-neutral', 'cited-note', 'live-note')").all();
      expect(rows).toEqual([{ note_id: "cited-note", update_count: 1, rule: 1 }]);
      const left = db.prepare("SELECT session_id FROM session_checkpoint ORDER BY session_id").all();
      expect(left).toEqual([{ session_id: "live-session" }]);
    } finally {
      db.close();
    }
  });
});
