/**
 * #37 review: Q must not be z-scored. With most candidates unlearned (Q = 0),
 * z-scoring gave one credited note +sqrt(n-1) whatever its value, so a Q of
 * 0.017 outranked the best text match.
 */
import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import { initQValueTables, updateQ } from "../../src/core/qvalue.js";
import { phaseB } from "../../src/core/rerank.js";

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

  it("a strongly learned note still gets a real, bounded lift", () => {
    for (let i = 0; i < 20; i++) updateQ(db, "note-5", 1.0, `s${i}`); // Q ~ 0.88
    const out = phaseB(db, cands(40), "q", "semantic", "s", {}, false) as { title: string }[];
    const pos = out.findIndex((r) => r.title === "note-5");
    // Moves up (measured: 5 -> 3) but does not jump the best matches.
    expect(pos).toBeLessThan(5);
    expect(out[0]!.title).toBe("note-0");
  });
});
