/**
 * #37 review: Q must not be z-scored. With most candidates unlearned (Q = 0),
 * z-scoring gave one credited note +sqrt(n-1) whatever its value, so a Q of
 * 0.017 outranked the best text match.
 */
import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import { initQValueTables, updateQ, incrementExposure } from "../../src/core/qvalue.js";
import { phaseB } from "../../src/core/rerank.js";
import { SessionRewardAccumulator } from "../../src/core/reward.js";
import { batchUpdateQ } from "../../src/core/qvalue.js";

let db: InstanceType<typeof Database>;
beforeEach(() => {
  db = new Database(":memory:");
  initQValueTables(db);
  // Past the lambda warm-up so Q is at full weight.
  for (let i = 0; i < 250; i++) updateQ(db, `warm-${i % 50}`, 0.3, "warm");
});

const cands = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ title: `note-${i}`, score: 1 - i * 0.02, signals: {} })) as never[];

describe("phaseB Q scaling", () => {
  it("one tiny credit does not lift a weak match above the best match", () => {
    updateQ(db, "note-30", 0.02, "s1"); // Q ~ 0.002 on a weak match
    const out = phaseB(db, cands(40), "q", "semantic", "s", {}, false) as { title: string }[];
    expect(out[0]!.title).toBe("note-0");
    expect(out.slice(0, 8).map((r) => r.title)).not.toContain("note-30");
  });

  it("a strongly learned note gets a real, bounded lift", () => {
    // Behaviour change, stated rather than hidden (Codex review): this test
    // used to require that a note with 20 citations stay below note-0 (5 -> 3).
    // That held only because the first credit cut the exploration bonus by
    // ~0.4, which offset most of Q - the #37 inversion. With it gone, a note
    // cited in all 20 of its showings rises to the top from 5 ranks down at
    // Q_SCALE = 0.5. Whether that is too strong is a tuning question (see
    // Q_SCALE); what this pins is the bound: Q adds at most LAMBDA_MAX /
    // Q_SCALE = 0.70, about 11 ranks at this spacing.
    for (let i = 0; i < 20; i++) incrementExposure(db, "note-5");
    for (let i = 0; i < 20; i++) updateQ(db, "note-5", 1.0, `s${i}`); // Q ~ 0.88
    const near = phaseB(db, cands(40), "q", "semantic", "s", { random: () => 1 }, false, undefined, { keepTail: true }) as { title: string }[];
    expect(near.findIndex((r) => r.title === "note-5")).toBe(0);

    for (let i = 0; i < 20; i++) incrementExposure(db, "note-20");
    for (let i = 0; i < 20; i++) updateQ(db, "note-20", 1.0, `t${i}`);
    const far = phaseB(db, cands(40), "q", "semantic", "s", { random: () => 1 }, false, undefined, { keepTail: true }) as { title: string }[];
    const pos = far.findIndex((r) => r.title === "note-20");
    expect(pos).toBeGreaterThanOrEqual(0);
    expect(pos).toBeLessThan(20); // a real lift
    expect(pos).toBeGreaterThan(5); // bounded: not to the top from 20 down
  });

  // Codex review of the #37 follow-up: at Q_SCALE = 2 this failed - a note
  // used 20 times scored 0.249 against 0.423 for a never-shown equal match.
  it("a heavily used note outranks an equally relevant note nobody has seen", () => {
    for (let i = 0; i < 20; i++) incrementExposure(db, "used");
    for (let i = 0; i < 20; i++) updateQ(db, "used", 1.0, `s${i}`);
    const tied = [
      { title: "unseen", score: 0.5, signals: {} },
      { title: "used", score: 0.5, signals: {} },
      ...Array.from({ length: 10 }, (_, i) => ({ title: `other-${i}`, score: 0.4 - i * 0.01, signals: {} })),
    ] as never[];
    const out = phaseB(db, tied, "q", "semantic", "s", { random: () => 1 }, false) as { title: string }[];
    const used = out.findIndex((r) => r.title === "used");
    expect(used).toBeGreaterThanOrEqual(0);
    expect(used).toBeLessThan(out.findIndex((r) => r.title === "unseen"));
  });
});

/**
 * #37 follow-up, measured on a 1,566-note vault before the fix: at equal
 * exposure a note's first credit LOWERED its phaseB score, because the UCB
 * bonus fell from c * 2.5 to the variance formula by more than the Q it added.
 * Two forward citations only broke even. Through the whole of phaseB, a use
 * must never move a note down.
 */
describe("phaseB: a use never moves a note down", () => {
  const spread = (n: number) =>
    Array.from({ length: n }, (_, i) => ({ title: `note-${i}`, score: 1 - i * 0.02, signals: {} })) as never[];

  /** Final position and score of note-4 when every candidate was shown 10 times. */
  function rankOf(credits: readonly number[]): { pos: number; score: number } {
    const d = new Database(":memory:");
    initQValueTables(d);
    for (let i = 0; i < 250; i++) updateQ(d, `warm-${i % 50}`, 0.3, "warm");
    for (let i = 0; i < 20; i++) for (let e = 0; e < 10; e++) incrementExposure(d, `note-${i}`);
    for (const r of credits) updateQ(d, "note-4", r, "s1");
    const out = phaseB(d, spread(20), "q", "semantic", "s", { random: () => 1 }, false) as { title: string; score: number }[];
    const pos = out.findIndex((r) => r.title === "note-4");
    return { pos, score: out[pos]?.score ?? -Infinity };
  }

  const base = () => rankOf([]);
  for (const [label, credits] of [
    ["one re-recall", [0.4]],
    ["one update", [0.5]],
    ["one forward citation", [1.0]],
    ["three forward citations", [1.0, 1.0, 1.0]],
  ] as const) {
    it(`${label}: scores higher than the same note uncredited, and ranks no lower`, () => {
      const b = base();
      const c = rankOf(credits);
      expect(c.score).toBeGreaterThan(b.score);
      expect(c.pos).toBeGreaterThanOrEqual(0);
      expect(c.pos).toBeLessThanOrEqual(b.pos);
    });
  }
});

// Codex review, round two: right after upgrading, legacy updates no longer
// count toward the lambda warm-up, so every vault starts at LAMBDA_MIN. At the
// old 0.15 a note with 20 citations and 20 exposures lost to a never-shown
// equal match (1.685 vs 1.693). The ranking invariant must hold from query one.
describe("warm-up: used beats unseen even with no other learning in the vault", () => {
  it("a heavily used note outranks an equally relevant unseen note at LAMBDA_MIN", () => {
    const fresh = new Database(":memory:");
    initQValueTables(fresh);
    for (let i = 0; i < 20; i++) incrementExposure(fresh, "used");
    for (let i = 0; i < 20; i++) updateQ(fresh, "used", 1.0, `s${i}`);
    const tied = [
      { title: "unseen", score: 0.5, signals: {} },
      { title: "used", score: 0.5, signals: {} },
      ...Array.from({ length: 10 }, (_, i) => ({ title: `other-${i}`, score: 0.4 - i * 0.01, signals: {} })),
    ] as never[];
    const out = phaseB(fresh, tied, "q", "semantic", "s", { random: () => 1 }, false) as { title: string }[];
    const used = out.findIndex((r) => r.title === "used");
    expect(used).toBeGreaterThanOrEqual(0);
    expect(used).toBeLessThan(out.findIndex((r) => r.title === "unseen"));
  });
});

// Codex review, round three: exposure itself cost bonus, so realistic use -
// shown, then used - still lost to a never-shown equal match. Only exposure
// NOT followed by use is damped now. These are the reviewer's two scenarios,
// on a fresh vault (LAMBDA_MIN) where the margin is smallest.
describe("used beats unseen across the realistic lifecycle", () => {
  function usedVsUnseen(exposures: number, credits: number[]): { used: number; unseen: number } {
    const fresh = new Database(":memory:");
    initQValueTables(fresh);
    for (let i = 0; i < exposures; i++) incrementExposure(fresh, "used");
    credits.forEach((r, i) => updateQ(fresh, "used", r, `s${i}`));
    const cands = [
      { title: "unseen", score: 0.5, signals: {} },
      { title: "used", score: 0.5, signals: {} },
      ...Array.from({ length: 38 }, (_, i) => ({ title: `other-${i}`, score: 0.45 - i * 0.01, signals: {} })),
    ] as never[];
    const out = phaseB(fresh, cands, "q", "semantic", "s", { random: () => 1 }, false, undefined, { keepTail: true }) as { title: string }[];
    const used = out.findIndex((r) => r.title === "used");
    const unseen = out.findIndex((r) => r.title === "unseen");
    // keepTail returns every candidate, so -1 would be a real bug, not a miss.
    expect(used).toBeGreaterThanOrEqual(0);
    expect(unseen).toBeGreaterThanOrEqual(0);
    return { used, unseen };
  }

  it("one exposure, one forward citation", () => {
    const r = usedVsUnseen(1, [1.0]);
    expect(r.used).toBeLessThan(r.unseen);
  });

  it("one exposure, one re-recall (the weakest credit)", () => {
    const r = usedVsUnseen(1, [0.4]);
    expect(r.used).toBeLessThan(r.unseen);
  });

  it("40 exposures, 20 re-recalls: used half the times it was shown", () => {
    const r = usedVsUnseen(40, Array(20).fill(0.4));
    expect(r.used).toBeLessThan(r.unseen);
  });
});

// Codex review, round four: the previous tests set exposure and credits by
// hand, in combinations production cannot produce (one exposure, one
// re-recall). These drive the real accumulator: each query shows the note
// (exposure), and credit comes from the session outcome.
describe("used beats unseen through real sessions", () => {
  type Session = { queries: string[]; cite?: boolean };
  function afterSessions(sessions: Session[]): { used: number; unseen: number } {
    const d = new Database(":memory:");
    initQValueTables(d);
    sessions.forEach((s, i) => {
      const acc = new SessionRewardAccumulator(`sess-${i}`);
      s.queries.forEach((q, rank) => {
        incrementExposure(d, "used");
        acc.logRetrieval("used", rank % 5, q, "semantic");
      });
      if (s.cite) acc.logAdd(`new-${i}`, "builds on [[used]]");
      batchUpdateQ(d, acc.computeRewards(d), `sess-${i}`);
    });
    const cands = [
      { title: "unseen", score: 0.5, signals: {} },
      { title: "used", score: 0.5, signals: {} },
      ...Array.from({ length: 38 }, (_, i) => ({ title: `other-${i}`, score: 0.45 - i * 0.01, signals: {} })),
    ] as never[];
    const out = phaseB(d, cands, "q", "semantic", "s", { random: () => 1 }, false, undefined, { keepTail: true }) as { title: string }[];
    const used = out.findIndex((r) => r.title === "used");
    const unseen = out.findIndex((r) => r.title === "unseen");
    // keepTail returns every candidate, so -1 would be a real bug, not a miss.
    expect(used).toBeGreaterThanOrEqual(0);
    expect(unseen).toBeGreaterThanOrEqual(0);
    return { used, unseen };
  }

  it("two distinct queries in one session (one re-recall credit)", () => {
    const r = afterSessions([{ queries: ["a", "b"] }]);
    expect(r.used).toBeLessThan(r.unseen);
  });

  it("four queries, then cited once", () => {
    const r = afterSessions([{ queries: ["a", "b", "c", "d"], cite: true }]);
    expect(r.used).toBeLessThan(r.unseen);
  });

  it("twenty sessions of five queries each, re-recalled every session", () => {
    const r = afterSessions(Array.from({ length: 20 }, () => ({ queries: ["a", "b", "c", "d", "e"] })));
    expect(r.used).toBeLessThan(r.unseen);
  });

  it("shown in many sessions and never used: exploration still damps it", () => {
    const r = afterSessions(Array.from({ length: 20 }, () => ({ queries: ["same"] })));
    expect(r.used).toBeGreaterThan(r.unseen);
  });
});
