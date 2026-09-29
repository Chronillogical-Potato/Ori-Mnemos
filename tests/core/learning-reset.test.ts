/**
 * #37: the reset is offered to the agent and applied only on the user's yes.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs, existsSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import Database from "better-sqlite3";
import { initQValueTables, getQ, updateQ, DEFAULT_Q } from "../../src/core/qvalue.js";
import {
  getLearningResetStatus,
  buildLearningResetNotice,
  applyLearningResetDecision,
  recordLearningResetShown,
  reapplyLearningResetAfterImport,
  countLegacyLearning,
  MIN_UPDATES_TO_OFFER,
  MAX_RESET_REMINDERS,
} from "../../src/core/learning-reset.js";

let dir: string;
let dbPath: string;
let db: InstanceType<typeof Database>;

/** Rows exactly as the pre-#37 code wrote them: first update from old_q 0.5. */
function seedLegacy(notes: number, updatesEach: number, prefix = "note"): void {
  const ins = db.prepare(
    "INSERT INTO q_history (note_id, old_q, new_q, reward, reward_source, session_id) VALUES (?,?,?,?,?,?)",
  );
  for (let n = 0; n < notes; n++) {
    let q = 0.5;
    for (let u = 0; u < updatesEach; u++) {
      const nq = q + 0.1 * (0 - q);
      ins.run(`${prefix}-${n}`, q, nq, 0, "session_batch", `cli-${n}-${u}`);
      q = nq;
    }
    db.prepare(
      `INSERT INTO note_q (note_id, q_value, update_count, exposure_count, reward_sum, reward_sq_sum)
       VALUES (?, ?, ?, ?, 0, 0)`,
    ).run(`${prefix}-${n}`, q, updatesEach, updatesEach + 2);
  }
}

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "ori-reset37-"));
  dbPath = path.join(dir, "embeddings.db");
  db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.exec("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)");
  initQValueTables(db);
  db.exec(`CREATE TABLE IF NOT EXISTS session_checkpoint (session_id TEXT PRIMARY KEY, rewards_json TEXT, updated_at TEXT)`);
});

afterEach(async () => {
  db.close();
  await fs.rm(dir, { recursive: true, force: true });
});

describe("learning reset offer", () => {
  it("does not ask on a vault with no learning", () => {
    const s = getLearningResetStatus(db);
    expect(s.offer).toBe(false);
    expect(buildLearningResetNotice(s)).toBeNull();
  });

  it("does not ask below the threshold", () => {
    seedLegacy(1, MIN_UPDATES_TO_OFFER - 1);
    expect(getLearningResetStatus(db).offer).toBe(false);
  });

  it("asks when old-rule learning is present, naming the tool and the counts", () => {
    seedLegacy(5, 6);
    const s = getLearningResetStatus(db);
    expect(s).toMatchObject({ legacyUpdates: 30, legacyNotes: 5, decision: null, offer: true });
    const notice = buildLearningResetNotice(s)!;
    expect(notice).toContain("ori_learning_reset");
    expect(notice).toContain("#37");
    expect(notice).toContain("30");
    expect(notice).toMatch(/do NOT call the tool/);
  });

  it("explains why, and offers a permanent no from the first ask", () => {
    seedLegacy(5, 6);
    const notice = buildLearningResetNotice(getLearningResetStatus(db))!;
    expect(notice).toMatch(/^Learning reset question, reminder 1 of 3\./);
    expect(notice).toContain("Why:");
    expect(notice).toMatch(/each time a note was used its score went down/);
    expect(notice).toMatch(/drown out anything learned under the fixed rule/);
    expect(notice).toMatch(/no and never ask again/);
    expect(notice).toMatch(/decision="declined"; that is permanent/);
  });

  it("asks at most 3 times, says so on the last one, then goes quiet", () => {
    seedLegacy(5, 6);
    const seen: string[] = [];
    for (let i = 0; i < MAX_RESET_REMINDERS + 2; i++) {
      const n = buildLearningResetNotice(getLearningResetStatus(db));
      if (n) {
        seen.push(n);
        recordLearningResetShown(db);
      }
    }
    expect(MAX_RESET_REMINDERS).toBe(3);
    expect(seen).toHaveLength(3);
    expect(seen[1]).toMatch(/reminder 2 of 3/);
    expect(seen[2]).toMatch(/reminder 3 of 3/);
    expect(seen[2]).toMatch(/last time I'll ask/);
    expect(seen[0]).not.toMatch(/last time/);
    expect(getLearningResetStatus(db)).toMatchObject({ timesShown: 3, offer: false, decision: null });
  });

  it("does not count learning written under the new rules", () => {
    for (let i = 0; i < 30; i++) updateQ(db, `fresh-${i % 5}`, 0.3, "s1");
    expect(getLearningResetStatus(db).legacyUpdates).toBe(0);
  });
});

describe("review findings (regressions)", () => {
  it("backup is a valid, complete copy while another connection holds a read snapshot", async () => {
    seedLegacy(5, 6);
    const reader = new Database(dbPath);
    reader.prepare("BEGIN").run();
    reader.prepare("SELECT COUNT(*) FROM q_history").get();
    seedLegacy(5, 6, "late"); // committed after the reader's snapshot: lives only in the WAL
    recordLearningResetShown(db);
    const r = applyLearningResetDecision(db, dbPath, "accepted");
    reader.prepare("COMMIT").run();
    reader.close();
    const bak = new Database(r.backup!, { readonly: true });
    try {
      expect((bak.prepare("PRAGMA integrity_check").get() as { integrity_check: string }).integrity_check).toBe("ok");
      expect((bak.prepare("SELECT COUNT(*) c FROM q_history").get() as { c: number }).c).toBe(60);
    } finally {
      bak.close();
    }
  });

  it("a second or unprompted accept does not wipe learning made under the fixed rules", () => {
    seedLegacy(5, 6);
    recordLearningResetShown(db);
    applyLearningResetDecision(db, dbPath, "accepted");
    for (let i = 0; i < 25; i++) updateQ(db, `fresh-${i % 5}`, 0.3, "s2");
    const again = applyLearningResetDecision(db, dbPath, "accepted");
    expect(again.skipped).toBe("already reset");
    expect(again.backup).toBeNull();
    expect((db.prepare("SELECT COUNT(*) c FROM note_q WHERE update_count > 0").get() as { c: number }).c).toBe(5);
  });

  it("an unprompted accept (never shown) is refused", () => {
    seedLegacy(5, 6);
    const r = applyLearningResetDecision(db, dbPath, "accepted");
    expect(r.skipped).toBe("not offered");
    expect((db.prepare("SELECT COUNT(*) c FROM note_q WHERE update_count > 0").get() as { c: number }).c).toBe(5);
  });

  it("the reset keeps learning made under the fixed rule", () => {
    seedLegacy(5, 6);
    for (let i = 0; i < 10; i++) updateQ(db, `fresh-${i % 2}`, 0.3, "s2");
    recordLearningResetShown(db);
    const r = applyLearningResetDecision(db, dbPath, "accepted");
    expect(r.notesCleared).toBe(5);
    expect(r.historyArchived).toBe(30);
    expect((db.prepare("SELECT COUNT(*) c FROM note_q WHERE update_count > 0").get() as { c: number }).c).toBe(2);
    expect((db.prepare("SELECT COUNT(*) c FROM q_history").get() as { c: number }).c).toBe(10);
  });

  it("accept on a vault with no old-rule learning is a no-op", () => {
    for (let i = 0; i < 25; i++) updateQ(db, `fresh-${i % 5}`, 0.3, "s2");
    recordLearningResetShown(db);
    recordLearningResetShown(db);
    const r = applyLearningResetDecision(db, dbPath, "accepted");
    expect(r.skipped).toBe("no old-rule learning");
    expect((db.prepare("SELECT COUNT(*) c FROM note_q WHERE update_count > 0").get() as { c: number }).c).toBe(5);
  });

  it("the shown counter is atomic and returns the new count", () => {
    expect(recordLearningResetShown(db)).toBe(1);
    expect(recordLearningResetShown(db)).toBe(2);
    expect(getLearningResetStatus(db).timesShown).toBe(2);
  });

  it("after a decline the status check does not scan history", () => {
    seedLegacy(5, 6);
    applyLearningResetDecision(db, dbPath, "declined");
    expect(getLearningResetStatus(db)).toMatchObject({ decision: "declined", offer: false, legacyUpdates: 0 });
  });
});

describe("import of pre-reset learning", () => {
  it("after a yes: old rows brought back are cleared again, without re-asking", () => {
    seedLegacy(5, 6);
    recordLearningResetShown(db);
    applyLearningResetDecision(db, dbPath, "accepted");
    const before = countLegacyLearning(db);
    seedLegacy(5, 6, "imported"); // what importLearned restores from an old export
    const r = reapplyLearningResetAfterImport(db, dbPath, before);
    expect(r).toMatchObject({ added: 5, cleared: true }); // one first-update (old_q = 0.5) row per note
    expect(r.backup && existsSync(r.backup)).toBe(true);
    expect((db.prepare("SELECT COUNT(*) c FROM note_q WHERE update_count > 0").get() as { c: number }).c).toBe(0);
    expect(getLearningResetStatus(db)).toMatchObject({ decision: "accepted", timesShown: 1, offer: false });
  });

  it("after a no: nothing changes and the no stands", () => {
    seedLegacy(5, 6);
    applyLearningResetDecision(db, dbPath, "declined");
    const before = countLegacyLearning(db);
    seedLegacy(5, 6, "imported");
    expect(reapplyLearningResetAfterImport(db, dbPath, before).cleared).toBe(false);
    expect(getLearningResetStatus(db)).toMatchObject({ decision: "declined", offer: false });
  });
});

describe("learning reset decision", () => {
  it("declined: changes nothing and stops asking", () => {
    seedLegacy(5, 6);
    const before = getQ(db, "note-0");
    const r = applyLearningResetDecision(db, dbPath, "declined");
    expect(r).toMatchObject({ decision: "declined", backup: null, notesCleared: 0 });
    expect(getQ(db, "note-0")).toBe(before);
    expect(getLearningResetStatus(db)).toMatchObject({ decision: "declined", offer: false });
  });

  it("accepted: backs up, archives history, clears learned state, keeps exposure", () => {
    seedLegacy(5, 6);
    recordLearningResetShown(db);
    const r = applyLearningResetDecision(db, dbPath, "accepted");
    expect(r.decision).toBe("accepted");
    expect(r.backup && existsSync(r.backup)).toBe(true);
    expect(r.notesCleared).toBe(5);
    expect(r.historyArchived).toBe(30);

    const n = (sql: string) => (db.prepare(sql).get() as { c: number }).c;
    expect(n("SELECT COUNT(*) c FROM q_history")).toBe(0);
    expect(n("SELECT COUNT(*) c FROM q_history_pre_37")).toBe(30);
    expect(n("SELECT COUNT(*) c FROM note_q WHERE update_count > 0")).toBe(0);
    expect(n("SELECT COUNT(*) c FROM note_q WHERE exposure_count > 0")).toBe(5);
    expect(getQ(db, "note-0")).toBe(DEFAULT_Q);
    expect(getLearningResetStatus(db)).toMatchObject({ decision: "accepted", offer: false, legacyUpdates: 0 });

    // The backup is a real copy of the pre-reset state.
    const bak = new Database(r.backup!, { readonly: true });
    try {
      expect((bak.prepare("SELECT COUNT(*) c FROM q_history").get() as { c: number }).c).toBe(30);
    } finally {
      bak.close();
    }
  });

  it("after a reset, a positive reward raises Q above an unlearned note", () => {
    seedLegacy(5, 6);
    recordLearningResetShown(db);
    applyLearningResetDecision(db, dbPath, "accepted");
    updateQ(db, "note-0", 0.2, "s2");
    expect(getQ(db, "note-0")).toBeGreaterThan(getQ(db, "never-seen"));
  });
});
