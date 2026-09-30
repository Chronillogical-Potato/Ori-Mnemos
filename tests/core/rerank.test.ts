import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import {
  zNormalize,
  computeLambda,
  phaseB,
  LAMBDA_MIN,
  LAMBDA_MAX,
  LAMBDA_MATURITY,
  MAX_CUMULATIVE_BIAS,
  K2,
} from "../../src/core/rerank.js";
import { initQValueTables, updateQ } from "../../src/core/qvalue.js";
import type { ScoredNote } from "../../src/core/ranking.js";

let db: Database.Database;

beforeEach(() => {
  db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  initQValueTables(db);
});

describe("zNormalize", () => {
  it("returns empty for empty input", () => {
    expect(zNormalize([])).toEqual([]);
  });

  it("normalizes to mean 0 and std 1", () => {
    const result = zNormalize([2, 4, 6, 8, 10]);
    const mean = result.reduce((a, b) => a + b, 0) / result.length;
    const std = Math.sqrt(
      result.reduce((a, b) => a + b * b, 0) / result.length,
    );
    expect(mean).toBeCloseTo(0, 10);
    expect(std).toBeCloseTo(1, 10);
  });

  it("handles constant values (std=0 → uses 1)", () => {
    const result = zNormalize([5, 5, 5]);
    expect(result).toEqual([0, 0, 0]);
  });
});

describe("computeLambda", () => {
  it("starts at LAMBDA_MIN with no Q updates", () => {
    expect(computeLambda(0)).toBeCloseTo(LAMBDA_MIN, 10);
  });

  it("reaches LAMBDA_MAX at maturity and holds there", () => {
    expect(computeLambda(LAMBDA_MATURITY)).toBeCloseTo(LAMBDA_MAX, 10);
    expect(computeLambda(LAMBDA_MATURITY * 50)).toBeCloseTo(LAMBDA_MAX, 10);
  });

  it("ramps linearly between min and max", () => {
    expect(computeLambda(LAMBDA_MATURITY / 2)).toBeCloseTo(
      LAMBDA_MIN + 0.5 * (LAMBDA_MAX - LAMBDA_MIN),
      10,
    );
  });

  // The old bound was a literal [0.1, 0.6] that did not track the constants,
  // so a configured shift could exceed it and be silently truncated -- which
  // is what happened to procedural's +0.15. Range must follow the constants.
  it("never escapes [LAMBDA_MIN, LAMBDA_MAX] for any update count", () => {
    for (const n of [-1e9, -1, 0, 1, 199, 200, 201, 1e9, Number.MAX_SAFE_INTEGER]) {
      const lambda = computeLambda(n);
      expect(lambda).toBeGreaterThanOrEqual(LAMBDA_MIN);
      expect(lambda).toBeLessThanOrEqual(LAMBDA_MAX);
    }
  });

  it("is monotonically non-decreasing in update count", () => {
    let prev = -Infinity;
    for (let n = 0; n <= 400; n += 25) {
      const lambda = computeLambda(n);
      expect(lambda).toBeGreaterThanOrEqual(prev);
      prev = lambda;
    }
  });

  // lambda-sweep.mjs measured term-coverage recall@5 declining monotonically
  // above 0.35 on 1,653 replayed queries: -0.0048 at 0.40 and -0.0262 at 0.60,
  // both with bootstrap CIs excluding zero. Nothing may raise lambda past the
  // measured optimum without redoing that sweep.
  it("caps at the lambda measured optimal on the real query log", () => {
    expect(LAMBDA_MAX).toBeLessThanOrEqual(0.35);
  });
});

describe("phaseB", () => {
  const candidates: ScoredNote[] = [
    { title: "note-a", score: 0.95, signals: { rrf: 0.95 } },
    { title: "note-b", score: 0.80, signals: { rrf: 0.80 } },
    { title: "note-c", score: 0.60, signals: { rrf: 0.60 } },
    { title: "note-d", score: 0.40, signals: { rrf: 0.40 } },
    { title: "note-e", score: 0.30, signals: { rrf: 0.30 } },
  ];

  it("returns empty for empty candidates", () => {
    expect(phaseB(db, [], "query", "semantic", "s1")).toEqual([]);
  });

  it("returns at most K2 results", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({
      title: `note-${i}`,
      score: 1 - i * 0.04,
      signals: { rrf: 1 - i * 0.04 },
    }));
    const result = phaseB(db, many, "query", "semantic", "s1");
    expect(result.length).toBeLessThanOrEqual(K2);
  });

  it("moves scores in the direction Q was learned", () => {
    // note-d: strongly positive Q. note-a: strongly negative.
    const scoreOf = (d: Database.Database) =>
      new Map(phaseB(d, candidates, "query", "procedural", "s1", { random: () => 1 }, false).map((r) => [r.title, r.score]));

    const baseDb = new Database(":memory:");
    initQValueTables(baseDb);
    const base = scoreOf(baseDb);

    for (let i = 0; i < 50; i++) updateQ(db, "note-d", 1.0, "s0");
    for (let i = 0; i < 50; i++) updateQ(db, "note-a", -0.5, "s0");
    const learned = scoreOf(db);

    expect(learned.get("note-d")!).toBeGreaterThan(base.get("note-d")!);
    expect(learned.get("note-a")!).toBeLessThan(base.get("note-a")!);

    // Deliberately NOT asserted: a rank flip. This fixture's candidates sit
    // ~0.45 blended units apart, and Q is a bounded nudge (see Q_SCALE in
    // rerank.ts). The old version of this test did observe a flip, but only
    // because a credited note lost its flat exploration bonus - the #37
    // inversion - which the previous comment here documented as "exploration
    // outweighs a fully-learned positive Q roughly 2:1".
  });

  it("keepTail returns every candidate, reranked, so the caller can filter then trim", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ title: `note-${i}`, score: 1 - i * 0.04, signals: {} }));
    const out = phaseB(db, many, "query", "semantic", "s1", { random: () => 1 }, false, undefined, { k: 10, keepTail: true });
    expect(out).toHaveLength(20);
    expect(new Set(out.map((r) => r.title)).size).toBe(20);
  });

  it("respects cumulative bias cap", () => {
    const result = phaseB(db, candidates, "query", "semantic", "s1");
    for (const r of result) {
      const original = candidates.find((c) => c.title === r.title);
      if (original) {
        // Score should not exceed MAX_CUMULATIVE_BIAS * original (before compression)
        // After compression it can be slightly above but controlled
        expect(r.score).toBeLessThan(
          original.score * MAX_CUMULATIVE_BIAS * 2,
        );
      }
    }
  });

  it("logs retrievals to retrieval_log", () => {
    phaseB(db, candidates, "test query", "semantic", "session-1");
    const rows = db.prepare("SELECT * FROM retrieval_log").all() as any[];
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0].session_id).toBe("session-1");
    expect(rows[0].query_text).toBe("test query");
  });

  it("increments exposure count for all candidates", () => {
    phaseB(db, candidates, "query", "semantic", "s1");
    const row = db
      .prepare("SELECT exposure_count FROM note_q WHERE note_id = ?")
      .get("note-a") as { exposure_count: number } | undefined;
    expect(row?.exposure_count).toBe(1);
  });
});
