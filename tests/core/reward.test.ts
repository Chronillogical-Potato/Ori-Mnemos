import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import {
  SessionRewardAccumulator,
} from "../../src/core/reward.js";
import {
  initQValueTables,
  incrementExposure,
  getQ,
  getQState,
  DEFAULT_Q,
} from "../../src/core/qvalue.js";

let db: Database.Database;

beforeEach(() => {
  db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  initQValueTables(db);
});

describe("SessionRewardAccumulator", () => {
  it("hasData returns false with no retrievals", () => {
    const acc = new SessionRewardAccumulator("s1");
    expect(acc.hasData()).toBe(false);
  });

  it("hasData returns true after logging a retrieval", () => {
    const acc = new SessionRewardAccumulator("s1");
    acc.logRetrieval("note-a", 0, "query", "semantic");
    expect(acc.hasData()).toBe(true);
  });

  describe("forward citation detection", () => {
    it("gives +1.0 reward when retrieved note is cited in ori_add content", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logAdd("new-note", "This builds on [[note-a]] and extends it");

      const rewards = acc.computeRewards(db);
      expect(rewards.get("note-a")).toBeCloseTo(1.0, 10);
    });

    it("handles multiple citations", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logRetrieval("note-b", 1, "query", "semantic");
      acc.logAdd("new-note", "Combines [[note-a]] and [[note-b]]");

      const rewards = acc.computeRewards(db);
      expect(rewards.get("note-a")).toBeCloseTo(1.0, 10);
      expect(rewards.get("note-b")).toBeCloseTo(1.0, 10);
    });

    it("does not give citation reward for non-retrieved notes", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logAdd("new-note", "References [[note-x]] which was not retrieved");

      const rewards = acc.computeRewards(db);
      // note-a was retrieved but not cited: no credit, even though a creation
      // happened (#37 review: downstream creation is reported, not credited).
      expect(rewards.has("note-a")).toBe(false);
    });
  });

  describe("update reward", () => {
    it("gives +0.5 reward when a retrieved note is updated", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logUpdate("note-a");

      const rewards = acc.computeRewards(db);
      expect(rewards.get("note-a")).toBeCloseTo(0.5, 10);
    });
  });

  describe("downstream creation (#37 review: reported, not credited)", () => {
    it("does not credit notes the new content does not link to", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logRetrieval("note-b", 2, "query", "semantic");
      acc.logAdd("new-note", "A new insight"); // no [[citation]]

      const rewards = acc.computeRewards(db);
      expect(rewards.size).toBe(0);
      expect(acc.getSignalCounts().downstream_creation).toBe(2);
    });
  });

  describe("re-recall (#37 review)", () => {
    it("needs distinct queries: a retry of the same query is not a re-recall", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "same query", "semantic");
      acc.logRetrieval("note-a", 0, "Same Query ", "semantic");
      expect(acc.computeRewards(db).has("note-a")).toBe(false);
    });
  });

  describe("dead end penalty", () => {
    it("gives negative reward for top-3 notes with no follow-up", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logRetrieval("note-b", 1, "query", "semantic");
      acc.logRetrieval("note-c", 3, "query", "semantic");

      const rewards = acc.computeRewards(db);
      // Top-3 with no follow-up: reported as dead_end, never credited (#37
      // review). Read-and-answered-from is indistinguishable from useless.
      expect(rewards.has("note-a")).toBe(false);
      expect(rewards.has("note-b")).toBe(false);
      expect(acc.getSignalCounts().dead_end).toBe(2);
      // Rank 3 (> 2): neutral, which is no signal and so no update at all
      // (#37). Writing it as reward 0 pulled Q toward 0 on every retrieval.
      expect(rewards.has("note-c")).toBe(false);
      expect(acc.getSignalCounts().neutral).toBe(1);
    });

    it("partial follow-up is reported but never lowers a learned Q (#37)", () => {
      const acc0 = new SessionRewardAccumulator("s0");
      acc0.logRetrieval("hub", 0, "q", "semantic");
      acc0.logRetrieval("hub", 0, "q2", "semantic"); // re-recall: +0.4
      acc0.concludeSession(db);
      const learned = getQ(db, "hub");
      // "other" is updated, nothing is created: hub falls to partial_follow_up.
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("hub", 1, "q", "semantic");
      acc.logRetrieval("other", 0, "q", "semantic");
      acc.logUpdate("other");
      const rewards = acc.computeRewards(db);
      expect(acc.getSignalCounts().partial_follow_up).toBe(1);
      expect(rewards.has("hub")).toBe(false);
      acc.concludeSession(db);
      expect(getQ(db, "hub")).toBe(learned);
    });

    it("never lowers Q of a note that was only read (#37)", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.concludeSession(db);
      expect(getQ(db, "note-a")).toBe(DEFAULT_Q);
    });
  });

  describe("no exposure correction on credit (#37)", () => {
    it("pays the full signal regardless of how often the note was shown", () => {
      // With an EMA, dividing credit by lifetime exposure made Q's ceiling fall
      // with use: cited every session, a note peaked at 0.48 and sank to 0.20.
      for (let i = 0; i < 300; i++) incrementExposure(db, "note-a");
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logAdd("new-note", "Extends [[note-a]]");
      expect(acc.computeRewards(db).get("note-a")!).toBeCloseTo(1.0, 10);
    });

    it("a note cited every session keeps climbing, and outranks one cited rarely", () => {
      const cite = (id: string, sid: string) => {
        const acc = new SessionRewardAccumulator(sid);
        acc.logRetrieval(id, 0, "q", "semantic");
        incrementExposure(db, id);
        acc.logAdd(`n-${sid}`, `cites [[${id}]]`);
        acc.concludeSession(db);
      };
      for (let s = 0; s < 40; s++) cite("hub", `h${s}`);
      for (let s = 0; s < 3; s++) cite("niche", `n${s}`);
      expect(getQ(db, "hub")).toBeGreaterThan(0.9);
      expect(getQ(db, "hub")).toBeGreaterThan(getQ(db, "niche"));
    });
  });

  describe("within-session re-recall", () => {
    it("gives diminishing reward for re-retrieved notes", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query1", "semantic");
      acc.logRetrieval("note-a", 1, "query2", "semantic");
      // No follow-up, but re-recalled (ranks.length > 1)

      const rewards = acc.computeRewards(db);
      // Constant since #37 (was 0.4 / count)
      expect(rewards.get("note-a")!).toBeCloseTo(0.4, 10);
    });
  });

  describe("reward clamping", () => {
    it("clamps rewards to [-1, 1]", () => {
      const acc = new SessionRewardAccumulator("s1");
      acc.logRetrieval("note-a", 0, "query", "semantic");
      acc.logAdd("new-note", "Uses [[note-a]]");

      const rewards = acc.computeRewards(db);
      for (const reward of rewards.values()) {
        expect(reward).toBeGreaterThanOrEqual(-1);
        expect(reward).toBeLessThanOrEqual(1);
      }
    });
  });
});

describe("concludeSession (fix list item 5)", () => {
  it("turns a retrieve-then-conclude session into non-zero update_count", () => {
    // The measured production state was 717 note_q rows with 707 at
    // update_count = 0: retrieval ran everywhere, the session flush only in
    // the MCP server, and nothing in between recorded that learning had never
    // happened. One call now closes the loop.
    const acc = new SessionRewardAccumulator("s1");
    acc.logRetrieval("note-a", 0, "q", "semantic");
    acc.logRetrieval("note-b", 1, "q", "semantic");
    acc.logRetrieval("note-c", 5, "q", "semantic");
    acc.logAdd("Synthesis", "building on [[note a]]");

    // Only the cited note is credited; b and c were retrieved alongside.
    expect(acc.concludeSession(db)).toBe(1);
    for (const id of ["note-a"]) {
      expect(getQState(db, id).updateCount).toBeGreaterThan(0);
      expect(getQState(db, id).learned).toBe(true);
    }
    // The cited note actually moved, so the credit is real and not a no-op
    // write that merely bumps the counter.
    expect(getQ(db, "note-a")).toBeGreaterThan(DEFAULT_Q);
  });

  it("writes through the sanctioned source only", () => {
    const acc = new SessionRewardAccumulator("s1");
    // Re-recalled, so it earns real credit (a lone top hit is a dead end,
    // which is reported but never credited since #37).
    acc.logRetrieval("note-a", 0, "q", "semantic");
    acc.logRetrieval("note-a", 0, "q2", "semantic");
    acc.concludeSession(db);
    const rows = db
      .prepare("SELECT DISTINCT reward_source FROM q_history")
      .all() as { reward_source: string }[];
    expect(rows.map((r) => r.reward_source)).toEqual(["session_batch"]);
  });

  it("refuses to flush twice", () => {
    // Rewards accumulate for the whole session and there is no clear(), so a
    // second flush would recompute over the same set and inflate update_count
    // and reward_sum on every pass.
    const acc = new SessionRewardAccumulator("s1");
    // Re-recalled, so it earns real credit (a lone top hit is a dead end,
    // which is reported but never credited since #37).
    acc.logRetrieval("note-a", 0, "q", "semantic");
    acc.logRetrieval("note-a", 0, "q2", "semantic");
    expect(acc.concludeSession(db)).toBe(1);
    expect(acc.isFlushed()).toBe(true);
    expect(acc.concludeSession(db)).toBe(0);
    expect(getQState(db, "note-a").updateCount).toBe(1);
  });

  it("reports zero when a session retrieved nothing", () => {
    const acc = new SessionRewardAccumulator("s1");
    acc.logAdd("Standalone", "no citations here");
    expect(acc.concludeSession(db)).toBe(0);
    expect(acc.isFlushed()).toBe(false);
    expect(db.prepare("SELECT COUNT(*) n FROM q_history").get()).toEqual({
      n: 0,
    });
  });

  it("accepts explore_conclude as the source for a navigated session", () => {
    const acc = new SessionRewardAccumulator("s1");
    // Re-recalled, so it earns real credit (a lone top hit is a dead end,
    // which is reported but never credited since #37).
    acc.logRetrieval("note-a", 0, "q", "semantic");
    acc.logRetrieval("note-a", 0, "q2", "semantic");
    expect(acc.concludeSession(db, "explore_conclude")).toBe(1);
    const rows = db
      .prepare("SELECT DISTINCT reward_source FROM q_history")
      .all() as { reward_source: string }[];
    expect(rows.map((r) => r.reward_source)).toEqual(["explore_conclude"]);
  });

  it("cannot launder an unsanctioned source", () => {
    const acc = new SessionRewardAccumulator("s1");
    // Re-recalled, so it earns real credit (a lone top hit is a dead end,
    // which is reported but never credited since #37).
    acc.logRetrieval("note-a", 0, "q", "semantic");
    acc.logRetrieval("note-a", 0, "q2", "semantic");
    expect(() => acc.concludeSession(db, "manual")).toThrow(
      /refusing write from source/,
    );
  });
});

// #37 follow-up (Codex review): downstream_creation became report-only in #37
// but was still checked before re_recall, so writing ANY note in a session
// erased every re-recall credit in it.
describe("credited signals win over report-only ones", () => {
  it("re-recall is credited even when the session also created an unrelated note", () => {
    const acc = new SessionRewardAccumulator("s1");
    acc.logRetrieval("used", 1, "first question", "semantic");
    acc.logRetrieval("used", 2, "a different question", "semantic");
    acc.logAdd("unrelated", "no links here");
    const rewards = acc.computeRewards(db);
    expect(rewards.get("used")).toBeCloseTo(0.4, 10);
    expect(acc.getSignalCounts()).toMatchObject({ re_recall: 1 });
  });
});
