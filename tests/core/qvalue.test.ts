import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import {
  initQValueTables,
  getQ,
  getDecayedQ,
  getRewardStats,
  getExposureCount,
  getTotalQUpdates,
  getTotalQueryCount,
  updateQ,
  incrementExposure,
  logRetrieval,
  explorationBonus,
  batchUpdateQ,
  getQState,
  exposureDamping,
  applyColdStartFloor,
  COLD_START_EPSILON,
  MIN_EXPLORE_RETENTION,
  ALPHA,
  DEFAULT_Q,
  getLearningHealth,
} from "../../src/core/qvalue.js";

let db: Database.Database;

beforeEach(() => {
  db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  initQValueTables(db);
});

describe("note_q.rule migration (#37 follow-up)", () => {
  it("classifies existing rows once: fixed-rule history -> 1, anything else -> legacy", () => {
    const d = new Database(":memory:");
    d.exec(`CREATE TABLE note_q (note_id TEXT PRIMARY KEY, q_value REAL NOT NULL DEFAULT 0,
      update_count INTEGER NOT NULL DEFAULT 0, exposure_count INTEGER NOT NULL DEFAULT 0,
      reward_sum REAL NOT NULL DEFAULT 0, reward_sq_sum REAL NOT NULL DEFAULT 0,
      last_updated TEXT NOT NULL DEFAULT (datetime('now')), last_reward REAL,
      created TEXT NOT NULL DEFAULT (datetime('now')));
      CREATE TABLE q_history (id INTEGER PRIMARY KEY AUTOINCREMENT, note_id TEXT NOT NULL,
      old_q REAL NOT NULL, new_q REAL NOT NULL, reward REAL NOT NULL, reward_source TEXT NOT NULL,
      session_id TEXT, timestamp TEXT NOT NULL DEFAULT (datetime('now')));`);
    const row = d.prepare("INSERT INTO note_q (note_id, q_value, update_count) VALUES (?, ?, ?)");
    const hist = d.prepare("INSERT INTO q_history (note_id, old_q, new_q, reward, reward_source) VALUES (?, ?, ?, 1, 'session_batch')");
    row.run("fresh", 0.1, 1); hist.run("fresh", 0, 0.1);           // fixed rule: starts at 0
    row.run("old", 0.45, 1); hist.run("old", 0.5, 0.45);           // old rule: starts at 0.5
    row.run("no-history", 0.407, 4);                               // history purged by an earlier reset
    row.run("unlearned", 0, 0);
    initQValueTables(d);
    const rules = Object.fromEntries((d.prepare("SELECT note_id, rule FROM note_q").all() as { note_id: string; rule: number }[]).map((r) => [r.note_id, r.rule]));
    expect(rules).toEqual({ fresh: 1, old: 0, "no-history": 0, unlearned: 0 });
    expect(getQState(d, "fresh")).toMatchObject({ learned: true, legacy: false });
    expect(getQState(d, "no-history")).toMatchObject({ learned: false, legacy: true });
    // Runs once: a later init does not reclassify.
    d.prepare("UPDATE note_q SET rule = 1 WHERE note_id = 'old'").run();
    initQValueTables(d);
    expect((d.prepare("SELECT rule FROM note_q WHERE note_id = 'old'").get() as { rule: number }).rule).toBe(1);
  });
});

describe("initQValueTables", () => {
  it("creates all three tables", () => {
    const tables = db
      .prepare(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
      )
      .all() as { name: string }[];
    const names = tables.map((t) => t.name);
    expect(names).toContain("note_q");
    expect(names).toContain("q_history");
    expect(names).toContain("retrieval_log");
  });

  it("is idempotent", () => {
    expect(() => initQValueTables(db)).not.toThrow();
  });
});

describe("getQ / updateQ", () => {
  it("returns DEFAULT_Q for unknown notes", () => {
    expect(getQ(db, "unknown-note")).toBe(DEFAULT_Q);
  });

  it("updates Q with EMA formula", () => {
    updateQ(db, "note-a", 1.0, "session-1");
    const q = getQ(db, "note-a");
    // Q = 0 + 0.1 * (1.0 - 0) = 0.1
    expect(q).toBeCloseTo(0.1, 10);
  });

  it("accumulates updates correctly", () => {
    updateQ(db, "note-a", 1.0, "s1");
    updateQ(db, "note-a", 1.0, "s1");
    const q = getQ(db, "note-a");
    // Round 1: 0 + 0.1*1.0*(1-0) = 0.1
    // Round 2: from the decayed 0.1 (seconds of decay, ~1e-8), + 0.1*1.0*(1-0.1) = 0.19
    expect(q).toBeCloseTo(0.19, 6);
  });

  // The #37 invariant: from the initial value, any positive reward raises Q
  // and any negative reward lowers it. Under DEFAULT_Q = 0.5 a reward of 0.3
  // (a good downstream_creation) lowered Q, so use was punished.
  it("moves Q in the direction of the reward's sign from the initial value", () => {
    for (const [id, r] of [["pos-small", 0.02], ["pos-typ", 0.3], ["neg", -0.05]] as const) {
      updateQ(db, id, r, "s1");
      expect(Math.sign(getQ(db, id))).toBe(Math.sign(r));
    }
  });

  it("getQState agrees with getQ on unlearned legacy rows", () => {
    db.prepare("INSERT INTO note_q (note_id, q_value, exposure_count) VALUES ('legacy-state', 0.5, 2)").run();
    expect(getQState(db, "legacy-state").q).toBe(getQ(db, "legacy-state"));
  });

  it("a credit never lowers Q, however small (#37 review)", () => {
    for (let i = 0; i < 12; i++) updateQ(db, "cited", 1.0, "s1");
    const before = getQ(db, "cited");
    updateQ(db, "cited", 0.4, "s2"); // a re-recall on a well-cited note
    expect(getQ(db, "cited")).toBeGreaterThanOrEqual(before);
  });

  it("an exposure-created row is not read as a learned value", () => {
    // Tables created before #37 give note_q.q_value a column default of 0.5.
    db.prepare("INSERT INTO note_q (note_id, q_value, exposure_count) VALUES ('legacy-row', 0.5, 3)").run();
    expect(getQ(db, "legacy-row")).toBe(DEFAULT_Q);
    updateQ(db, "legacy-row", 1.0, "s1");
    expect(getQ(db, "legacy-row")).toBeCloseTo(0.1, 10); // started from DEFAULT_Q, not 0.5
  });

  it("decreases Q for negative rewards", () => {
    updateQ(db, "note-a", -0.15, "s1");
    const q = getQ(db, "note-a");
    // Q = 0 + 0.1*(-0.15-0) = -0.015
    expect(q).toBeCloseTo(-0.015, 10);
  });

  it("writes to q_history", () => {
    updateQ(db, "note-a", 1.0, "session-1");
    const history = db
      .prepare("SELECT * FROM q_history WHERE note_id = ?")
      .all("note-a") as any[];
    expect(history).toHaveLength(1);
    expect(history[0].old_q).toBeCloseTo(0, 10);
    expect(history[0].new_q).toBeCloseTo(0.1, 10);
    expect(history[0].session_id).toBe("session-1");
  });
});

describe("getDecayedQ", () => {
  it("returns DEFAULT_Q for unknown notes", () => {
    expect(getDecayedQ(db, "unknown")).toBe(DEFAULT_Q);
  });

  it("returns current Q for recently updated notes", () => {
    updateQ(db, "note-a", 1.0, "s1");
    // Just updated — daysSince ≈ 0, decay ≈ 1.0
    const decayed = getDecayedQ(db, "note-a");
    expect(decayed).toBeCloseTo(0.1, 2);
  });
});

describe("getRewardStats", () => {
  it("returns defaults for unknown notes", () => {
    const stats = getRewardStats(db, "unknown");
    expect(stats.mean).toBe(0);
    expect(stats.variance).toBe(0.25);
    expect(stats.count).toBe(0);
  });

  it("computes mean and variance after updates", () => {
    updateQ(db, "note-a", 1.0, "s1");
    updateQ(db, "note-a", 0.5, "s1");
    const stats = getRewardStats(db, "note-a");
    expect(stats.count).toBe(2);
    expect(stats.mean).toBeCloseTo(0.75, 10);
    // variance = (1^2+0.5^2)/2 - 0.75^2 = 0.625 - 0.5625 = 0.0625
    expect(stats.variance).toBeCloseTo(0.0625, 10);
  });
});

describe("exposure", () => {
  it("starts at 0", () => {
    expect(getExposureCount(db, "note-a")).toBe(0);
  });

  it("increments correctly", () => {
    incrementExposure(db, "note-a");
    expect(getExposureCount(db, "note-a")).toBe(1);
    incrementExposure(db, "note-a");
    expect(getExposureCount(db, "note-a")).toBe(2);
  });
});

describe("getTotalQUpdates", () => {
  it("sums update counts across all notes", () => {
    expect(getTotalQUpdates(db)).toBe(0);
    updateQ(db, "note-a", 1.0, "s1");
    updateQ(db, "note-b", 0.5, "s1");
    expect(getTotalQUpdates(db)).toBe(2);
  });
});

describe("getTotalQueryCount", () => {
  it("counts distinct session+query pairs", () => {
    expect(getTotalQueryCount(db)).toBe(0);
    logRetrieval(db, "s1", "query1", "semantic", "note-a", 0, 0.9, 0.5, 0.1, 0.8);
    logRetrieval(db, "s1", "query1", "semantic", "note-b", 1, 0.8, 0.5, 0.1, 0.7);
    logRetrieval(db, "s1", "query2", "semantic", "note-a", 0, 0.9, 0.5, 0.1, 0.8);
    expect(getTotalQueryCount(db)).toBe(2); // 2 distinct queries
  });
});

describe("explorationBonus", () => {
  it("returns c * 2.5 for new notes (count=0)", () => {
    const bonus = explorationBonus({ mean: 0, variance: 0.25, count: 0 }, 100);
    expect(bonus).toBeCloseTo(0.2 * 2.5, 10);
  });

  // #37 follow-up. These used to assert the UCB-Tuned property "fewer credits,
  // bigger bonus". That property is what kept the inversion alive after the
  // DEFAULT_Q fix: the first credit roughly halved the bonus while adding only
  // ALPHA * reward of Q, so a used note ranked below an equally-shown unused
  // one. The bonus now depends on exposure alone.
  it("a credited note keeps the full bonus however often it was shown", () => {
    const full = explorationBonus({ mean: 0, variance: 0.25, count: 0, exposure: 0 }, 0);
    for (const exposure of [1, 10, 100, 10_000]) {
      expect(explorationBonus({ mean: 0.4, variance: 0, count: 1, exposure }, 0)).toBeCloseTo(full, 12);
      // Never-credited notes are still damped by exposure (fix-list item 8).
      expect(explorationBonus({ mean: 0, variance: 0.25, count: 0, exposure }, 0)).toBeLessThan(full);
    }
  });

  it("a credit never lowers a note's total boost (Q + bonus) at equal exposure", () => {
    const boost = (credits: number[]): number => {
      const d = new Database(":memory:");
      initQValueTables(d);
      for (let i = 0; i < 10; i++) incrementExposure(d, "n");
      for (const r of credits) updateQ(d, "n", r, "s");
      return getDecayedQ(d, "n") + explorationBonus(getRewardStats(d, "n"), 1700);
    };
    const none = boost([]);
    for (const credits of [[0.4], [0.5], [1], [1, 1], [0.4, 0.5, 1]]) {
      expect(boost(credits)).toBeGreaterThan(none);
    }
  });
});

describe("batchUpdateQ", () => {
  it("updates multiple notes in a transaction", () => {
    const rewards = new Map([
      ["note-a", 1.0],
      ["note-b", -0.15],
      ["note-c", 0.5],
    ]);
    batchUpdateQ(db, rewards, "session-1");

    expect(getQ(db, "note-a")).toBeCloseTo(0.1, 10);
    expect(getQ(db, "note-b")).toBeCloseTo(-0.015, 10);
    expect(getQ(db, "note-c")).toBeCloseTo(0.05, 10);
  });
});

describe("logRetrieval", () => {
  it("writes to retrieval_log", () => {
    logRetrieval(db, "s1", "test query", "semantic", "note-a", 0, 0.9, 0.5, 0.1, 0.8);
    const rows = db.prepare("SELECT * FROM retrieval_log").all() as any[];
    expect(rows).toHaveLength(1);
    expect(rows[0].session_id).toBe("s1");
    expect(rows[0].query_text).toBe("test query");
    expect(rows[0].note_id).toBe("note-a");
    expect(rows[0].rank).toBe(0);
  });
});

describe("getQState (fix list item 5)", () => {
  it("distinguishes a never-updated Q from a learned one at the same value", () => {
    // A reward of exactly DEFAULT_Q leaves the EMA where it started:
    // 0.5 + 0.1*(0.5 - 0.5) = 0.5. So this note has genuinely learned, and its
    // Q-value is indistinguishable from the initialisation constant.
    updateQ(db, "learned-note", DEFAULT_Q, "s1");
    // This one was only ever shown. `incrementExposure` creates the row with
    // the default q_value — the shape 707 of 717 production rows were in.
    incrementExposure(db, "shown-note");

    // The value alone cannot tell them apart. This is the defect.
    expect(getQ(db, "learned-note")).toBe(DEFAULT_Q);
    expect(getQ(db, "shown-note")).toBe(DEFAULT_Q);

    expect(getQState(db, "learned-note").learned).toBe(true);
    expect(getQState(db, "learned-note").updateCount).toBe(1);
    expect(getQState(db, "shown-note").learned).toBe(false);
    expect(getQState(db, "shown-note").updateCount).toBe(0);
  });

  it("reports an absent note as unlearned rather than as a 0.5 score", () => {
    const state = getQState(db, "no-such-note");
    expect(state.learned).toBe(false);
    expect(state.q).toBe(DEFAULT_Q);
    expect(state.decayedQ).toBe(DEFAULT_Q);
    expect(state.exposureCount).toBe(0);
    expect(state.lastUpdated).toBeNull();
  });

  it("does not decay a value that was never learned", () => {
    incrementExposure(db, "shown-note");
    db.prepare("UPDATE note_q SET last_updated = '2020-01-01 00:00:00'").run();
    // Decaying an un-updated row would manufacture a difference between two
    // notes that have both learned nothing, purely from when exposure created
    // the row.
    expect(getQState(db, "shown-note").decayedQ).toBe(DEFAULT_Q);
    expect(getQState(db, "shown-note").learned).toBe(false);
  });

  it("canonicalizes the key like every other read", () => {
    updateQ(db, "Some Note Title", 1.0, "s1");
    expect(getQState(db, "some-note-title").learned).toBe(true);
    expect(getQState(db, "Some Note Title").noteId).toBe("some-note-title");
  });

  it("carries learned and exposure through getRewardStats", () => {
    incrementExposure(db, "shown-note");
    incrementExposure(db, "shown-note");
    const unlearned = getRewardStats(db, "shown-note");
    expect(unlearned.learned).toBe(false);
    expect(unlearned.exposure).toBe(2);

    updateQ(db, "shown-note", 1.0, "s1");
    const learned = getRewardStats(db, "shown-note");
    expect(learned.learned).toBe(true);
    expect(learned.exposure).toBe(2);
    expect(learned.count).toBe(1);
  });
});

describe("exposure-aware exploration (fix list item 8)", () => {
  it("damps nothing when exposure is unknown", () => {
    // Backward compatibility: callers that pass no exposure get the old value.
    expect(exposureDamping(0)).toBe(1);
    expect(explorationBonus({ mean: 0, variance: 0.25, count: 0 }, 100)).toBe(
      explorationBonus(
        { mean: 0, variance: 0.25, count: 0, exposure: 0 },
        100,
      ),
    );
  });

  it("gives a never-surfaced note a strictly larger bonus than a saturated one", () => {
    // The production degeneracy: 707 of 717 rows had count=0, so every
    // candidate received the identical c*2.5 and the term cancelled out of the
    // ranking entirely.
    const cold = explorationBonus(
      { mean: 0, variance: 0.25, count: 0, exposure: 0 },
      500,
    );
    const hot = explorationBonus(
      { mean: 0, variance: 0.25, count: 0, exposure: 200 },
      500,
    );
    expect(cold).toBeGreaterThan(hot);
    expect(hot).toBeGreaterThan(0);
  });

  it("decreases with exposure down to a floor, and never to zero", () => {
    const bonuses = [0, 1, 10, 100].map((exposure) =>
      explorationBonus({ mean: 0, variance: 0.25, count: 0, exposure }, 500),
    );
    for (let i = 1; i < bonuses.length; i++) {
      expect(bonuses[i]!).toBeLessThan(bonuses[i - 1]!);
    }
    // Beyond ~225 exposures the damping is floored, so even the most saturated
    // note in a vault keeps a bonus. The term is a gradient, never an
    // exclusion — the same reason the stage bandit floors its epsilon.
    const saturated = explorationBonus(
      { mean: 0, variance: 0.25, count: 0, exposure: 100_000 },
      500,
    );
    expect(saturated).toBeCloseTo(0.2 * 2.5 * MIN_EXPLORE_RETENTION, 10);
    expect(saturated).toBeLessThan(bonuses.at(-1)!);
  });
});

describe("applyColdStartFloor (fix list item 8)", () => {
  const ranked = Array.from({ length: 12 }, (_, i) => ({
    title: `note-${i}`,
    score: 1 - i * 0.05,
  }));

  beforeEach(() => {
    // note-0..note-8 have been surfaced; note-9..note-11 never have.
    for (let i = 0; i <= 8; i++) incrementExposure(db, `note-${i}`);
  });

  it("promotes the best never-surfaced candidate when epsilon fires", () => {
    const top = applyColdStartFloor(db, ranked, 8, { random: () => 0 });
    expect(top).toHaveLength(8);
    // note-9 is the strongest note nobody has seen; it takes the weakest slot.
    expect(top.map((r) => r.title)).toContain("note-9");
    expect(top[7]!.title).toBe("note-9");
    // The head of the list is never disturbed.
    expect(top.slice(0, 7).map((r) => r.title)).toEqual(
      ranked.slice(0, 7).map((r) => r.title),
    );
  });

  it("returns the plain cut when epsilon does not fire", () => {
    const top = applyColdStartFloor(db, ranked, 8, { random: () => 1 });
    expect(top.map((r) => r.title)).toEqual(
      ranked.slice(0, 8).map((r) => r.title),
    );
  });

  it("is disabled by epsilon 0", () => {
    const top = applyColdStartFloor(db, ranked, 8, {
      epsilon: 0,
      random: () => 0,
    });
    expect(top.map((r) => r.title)).toEqual(
      ranked.slice(0, 8).map((r) => r.title),
    );
  });

  it("skips an already-surfaced candidate below the cut", () => {
    // note-8 is below the cut but has been shown, so it is not a cold note and
    // must not be promoted by the exploration floor.
    const top = applyColdStartFloor(db, ranked.slice(0, 10), 8, {
      random: () => 0,
    });
    expect(top[7]!.title).toBe("note-9");
  });

  it("leaves the cut alone when nothing below it is cold", () => {
    for (let i = 9; i <= 11; i++) incrementExposure(db, `note-${i}`);
    const top = applyColdStartFloor(db, ranked, 8, { random: () => 0 });
    expect(top.map((r) => r.title)).toEqual(
      ranked.slice(0, 8).map((r) => r.title),
    );
  });

  it("consumes its random draw before any early exit", () => {
    // The 2026-09-12 starvation bug was a short-circuit above the exploration
    // check. Drawing first also keeps a caller's random stream independent of
    // the candidate list, so behaviour does not change with vault size.
    let draws = 0;
    const random = () => {
      draws++;
      return 0;
    };
    applyColdStartFloor(db, ranked.slice(0, 3), 8, { random });
    applyColdStartFloor(db, [], 8, { random });
    expect(draws).toBe(2);
  });

  it("defaults to a 10% floor", () => {
    expect(COLD_START_EPSILON).toBe(0.1);
    const fires = applyColdStartFloor(db, ranked, 8, {
      random: () => COLD_START_EPSILON - 1e-9,
    });
    const misses = applyColdStartFloor(db, ranked, 8, {
      random: () => COLD_START_EPSILON,
    });
    expect(fires.map((r) => r.title)).toContain("note-9");
    expect(misses.map((r) => r.title)).not.toContain("note-9");
  });
});

// #37 follow-up: Q learned under the pre-#37 rules (first update from
// old_q = 0.5) is ignored at read time instead of deleted.
describe("legacy (pre-#37) learning", () => {
  /** A note as the old rules left it: history starting at 0.5, Q pulled below. */
  function seedLegacyNote(id: string, q = 0.38, updates = 3): void {
    db.prepare(
      `INSERT INTO note_q (note_id, q_value, update_count, exposure_count, reward_sum, reward_sq_sum, last_updated)
       VALUES (?, ?, ?, 7, 0, 0, datetime('now'))`,
    ).run(id, q, updates);
    let old = 0.5;
    for (let i = 0; i < updates; i++) {
      const next = old - 0.04;
      db.prepare(
        "INSERT INTO q_history (note_id, old_q, new_q, reward, reward_source) VALUES (?, ?, ?, 0, 'session_batch')",
      ).run(id, old, next);
      old = next;
    }
  }

  it("reads as unlearned everywhere ranking looks", () => {
    seedLegacyNote("old-note");
    expect(getQ(db, "old-note")).toBe(DEFAULT_Q);
    expect(getDecayedQ(db, "old-note")).toBe(DEFAULT_Q);
    expect(getRewardStats(db, "old-note")).toMatchObject({ learned: false, count: 0, exposure: 7 });
    expect(getQState(db, "old-note")).toMatchObject({ learned: false, legacy: true, q: DEFAULT_Q });
  });

  it("does not count toward the lambda warm-up", () => {
    seedLegacyNote("old-note", 0.38, 40);
    updateQ(db, "new-note", 1.0, "s");
    expect(getTotalQUpdates(db)).toBe(1);
  });

  it("is kept on disk, not deleted, until the note earns new credit", () => {
    seedLegacyNote("old-note", 0.38, 3);
    const row = db.prepare("SELECT q_value, update_count FROM note_q WHERE note_id = 'old-note'").get();
    expect(row).toEqual({ q_value: 0.38, update_count: 3 });
    expect((db.prepare("SELECT COUNT(*) n FROM q_history WHERE note_id = 'old-note'").get() as { n: number }).n).toBe(3);
  });

  it("first new credit archives the legacy history and starts from 0", () => {
    seedLegacyNote("old-note", 0.38, 3);
    updateQ(db, "old-note", 1.0, "s");
    expect(getQ(db, "old-note")).toBeCloseTo(ALPHA * 1.0, 10);
    expect(getQState(db, "old-note")).toMatchObject({ learned: true, legacy: false, updateCount: 1 });
    const archived = db.prepare("SELECT COUNT(*) n FROM q_history_pre_37 WHERE note_id = 'old-note'").get() as { n: number };
    expect(archived.n).toBe(3);
    const history = db.prepare("SELECT old_q, new_q FROM q_history WHERE note_id = 'old-note'").all();
    expect(history).toEqual([{ old_q: 0, new_q: ALPHA }]);
    // Reward sums restart with the new rule, so UCB stats are not a mix of both.
    expect(getRewardStats(db, "old-note")).toMatchObject({ count: 1, mean: 1 });
  });

  // import-learned REPLACES note_q rows but APPENDS q_history, so history
  // cannot say which rule produced the current value; the row's own marker can
  // (Codex review: one reproduction for each direction).
  it("an imported fixed-rule row stays valid even when legacy history is present", () => {
    seedLegacyNote("mixed", 0.38, 3);
    // What import writes for a row exported after retirement: its rule travels.
    db.prepare("UPDATE note_q SET q_value = 0.19, update_count = 2, rule = 1 WHERE note_id = 'mixed'").run();
    expect(getQState(db, "mixed")).toMatchObject({ legacy: false, learned: true, q: 0.19 });
    updateQ(db, "mixed", 1.0, "s");
    // Not retired: continues from 0.19 instead of being wiped to 0.
    expect(getQ(db, "mixed")).toBeCloseTo(0.19 + ALPHA * (1 - 0.19), 2);
  });

  it("an old export imported over fixed-rule learning is treated as legacy", () => {
    updateQ(db, "n", 1.0, "s"); // fresh learning, rule = 1
    // Old exports have no `rule`: import writes NULL along with the old value.
    db.prepare("UPDATE note_q SET q_value = 0.38, update_count = 3, rule = NULL WHERE note_id = 'n'").run();
    expect(getQState(db, "n")).toMatchObject({ legacy: true, learned: false, q: DEFAULT_Q });
    expect(getTotalQUpdates(db)).toBe(0);
  });

  it("legacy citations do not count in health's forward-citation figure", () => {
    seedLegacyNote("old-note", 0.38, 3);
    db.prepare("UPDATE q_history SET reward = 1.0 WHERE note_id = 'old-note'").run();
    updateQ(db, "new-note", 0.5, "s");
    const h = getLearningHealth(db);
    expect(h.forwardCitations).toBe(0);
    expect(h.totalUpdates).toBe(1);
    expect(h.legacyNotes).toBe(1);
  });

  it("leaves fixed-rule notes alone", () => {
    updateQ(db, "new-note", 1.0, "s");
    updateQ(db, "new-note", 1.0, "s");
    expect(getQState(db, "new-note")).toMatchObject({ learned: true, legacy: false, updateCount: 2 });
    expect(db.prepare("SELECT name FROM sqlite_master WHERE name = 'q_history_pre_37'").get()).toBeUndefined();
  });
});
