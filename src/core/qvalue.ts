/**
 * Q-value storage, update, decay, and exploration bonus.
 * Layer 1 of retrieval intelligence — learns which notes are useful
 * via exponential moving average Q-updates with an exposure-damped exploration bonus.
 *
 * Research: MemRL, Drift, Tempera, bandit theory (63-source synthesis)
 *
 * ## Two invariants, both added 2026-08-28 after a production postmortem
 *
 * **Canonical keys.** Every `noteId` crossing this module is normalized with
 * `slugify()`. Before this, `retrieval_log` on a live vault held 1,072 distinct
 * ids — 772 slugs and 300 raw titles — for what were often the same notes, so
 * Q-values were split across two spellings and citation matching in reward.ts
 * could never resolve. Normalizing at the storage boundary means callers may
 * pass either form.
 *
 * **Sourced writes.** `updateQ` requires an explicit `RewardSource`. It used to
 * hardcode `'session_batch'`, which made every row look like deliberate
 * session-end credit — including the 12,682 rows written by a per-query rank
 * proxy in serve.ts that drowned the real signal at 93.4% of all history.
 * `assertSessionFlush` additionally refuses non-session-end writes unless the
 * caller opts in explicitly, so the same mistake cannot be made silently again.
 */

import type Database from "better-sqlite3";
import { slugify } from "./slug.js";

// Constants
const ALPHA = 0.1;
/**
 * Initial Q, and the value an unlearned note reports. 0 is the neutral point of
 * the reward scale (#37): nearly every reward the accumulator paid lay in
 * [-0.15, 0.6], and at the old 0.5 an EMA toward any typical reward was a
 * decrease, so retrieval -> update -> lower Q regardless of usefulness.
 * Measured on a 1,565-note vault: 3,315 of 3,320 updates lowered Q and every
 * learned note sat below every never-retrieved one.
 */
const DEFAULT_Q = 0;
const DECAY_RATE = 0.007; // half-life ~99 days
const SLOW_DECAY_Q = 0.3; // well-credited notes keep their value longer
const FAST_DECAY_Q = 0.05; // barely-credited notes fade faster
const EXPOSURE_BETA = 0.5;

/**
 * Exposure damping exponent for the exploration bonus (fix list item 8).
 * At 0.35 a note shown 10 times keeps 46% of its bonus and one shown 100 times
 * keeps 20% - a real gradient toward cold notes without erasing the bonus for
 * anything popular.
 */
const EXPLORE_EXPOSURE_BETA = 0.35;

/**
 * Floor on that damping. An over-exposed note still carries some exploration
 * bonus, so the term can never become a hard exclusion: the same reason the
 * stage bandit keeps an epsilon floor (see docs/stage-bandit-starvation.md).
 */
const MIN_EXPLORE_RETENTION = 0.15;

// --- Schema ---

export function initQValueTables(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS note_q (
      note_id TEXT PRIMARY KEY,
      q_value REAL NOT NULL DEFAULT 0,
      update_count INTEGER NOT NULL DEFAULT 0,
      exposure_count INTEGER NOT NULL DEFAULT 0,
      reward_sum REAL NOT NULL DEFAULT 0,
      reward_sq_sum REAL NOT NULL DEFAULT 0,
      last_updated TEXT NOT NULL DEFAULT (datetime('now')),
      last_reward REAL,
      created TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS q_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      note_id TEXT NOT NULL,
      old_q REAL NOT NULL,
      new_q REAL NOT NULL,
      reward REAL NOT NULL,
      reward_source TEXT NOT NULL,
      session_id TEXT,
      timestamp TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS retrieval_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      query_text TEXT NOT NULL,
      query_type TEXT,
      note_id TEXT NOT NULL,
      rank INTEGER NOT NULL,
      similarity_score REAL,
      q_score REAL,
      ucb_bonus REAL,
      final_score REAL,
      timestamp TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_q_history_note ON q_history(note_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_session ON retrieval_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_note ON retrieval_log(note_id);
  `);
  migrateRuleColumn(db);
}

/**
 * Add `note_q.rule` and classify the rows that existed before it.
 *
 * `rule = 1` marks Q learned under the fixed #37 rules; only `updateQ` writes
 * it. Anything else with `update_count > 0` - 0.7.0 rows, and rows restored by
 * `import-learned` from an export that predates the column (NULL) - is legacy.
 * A marker on the row, not an inference from q_history, because import
 * REPLACES note_q rows but APPENDS q_history: history cannot say which rule
 * produced the value currently in the row (Codex review, two reproductions).
 *
 * One-time classification of existing rows: fixed-rule learning always begins
 * with a transition from exactly 0 (DEFAULT_Q), which the old EMA from 0.5
 * never produces, so a row is fixed-rule if its history has such a transition
 * and no old-rule start. Unreleased builds are the only source of those rows.
 * The ALTER runs inside the transaction, so of two processes opening the same
 * vault only one can classify.
 *
 * Exported because some entry points open the database without
 * initQValueTables: `ori health` (the new queries need the column) and
 * `index export-learned` / `import-learned`, where importing into an
 * unmigrated table would silently drop incoming `rule` values and then let
 * this classification re-infer them from mixed history (Codex review).
 */
export function migrateRuleColumn(db: Database.Database): void {
  const cols = db.prepare("PRAGMA table_info(note_q)").all() as { name: string }[];
  // No note_q yet: nothing to migrate, and not this function's job to create it
  // (import-learned deliberately refuses to invent tables).
  if (cols.length === 0 || cols.some((c) => c.name === "rule")) return;
  try {
    db.transaction(() => {
      db.exec("ALTER TABLE note_q ADD COLUMN rule INTEGER DEFAULT 0");
      // No history, nothing to classify from: every learned row stays legacy.
      const hasHistory = db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'q_history'").get();
      if (hasHistory) db.exec(`
        UPDATE note_q SET rule = 1
         WHERE update_count > 0
           AND EXISTS (SELECT 1 FROM q_history h WHERE h.note_id = note_q.note_id AND h.old_q = 0)
           AND NOT EXISTS (SELECT 1 FROM q_history h WHERE h.note_id = note_q.note_id AND h.old_q = 0.5)
      `);
    })();
  } catch (err) {
    // Another process added it between the check and the ALTER.
    if (!/duplicate column/i.test(String(err))) throw err;
  }
}

// --- Legacy (pre-#37) learning ---

/**
 * Notes whose Q was learned under the pre-#37 rules: learned, and not marked
 * `rule = 1` (see migrateRuleColumn).
 *
 * These values are IGNORED at read time rather than deleted. Measured on a
 * 1,566-note vault: 1,085 legacy notes averaging Q = 0.373, built from 3,320
 * updates of which not one was a forward citation or an update signal - so
 * they encode how often a note was retrieved, not whether it helped. On the
 * fixed scale one forward citation is worth 0.1, so leaving them live let the
 * noise outrank every new credit, for every user who declined or never saw
 * the opt-in reset. The rows and history stay on disk; the opt-in reset can
 * still archive them.
 */
export const LEGACY_NOTES_SQL =
  "SELECT note_id FROM note_q WHERE update_count > 0 AND COALESCE(rule, 0) = 0";

/** SQL expression, true when note_q row `alias` holds legacy learning. */
const IS_LEGACY = (alias: string) =>
  `(${alias}.update_count > 0 AND COALESCE(${alias}.rule, 0) = 0)`;

/**
 * Move one note's legacy history into `q_history_pre_37` and clear its learned
 * columns, so a new credit starts from 0 instead of on top of legacy state.
 * Called by `updateQ` the first time a legacy note earns fixed-rule credit.
 * Same archive table and columns as the opt-in reset in learning-reset.ts.
 */
function retireLegacyNote(db: Database.Database, noteId: string): void {
  db.transaction(() => {
    db.exec(`
      CREATE TABLE IF NOT EXISTS q_history_pre_37 (
        id INTEGER, note_id TEXT, old_q REAL, new_q REAL,
        reward REAL, reward_source TEXT, session_id TEXT, timestamp TEXT
      )
    `);
    db.prepare(
      `INSERT INTO q_history_pre_37
         SELECT id, note_id, old_q, new_q, reward, reward_source, session_id, timestamp
           FROM q_history WHERE note_id = ?`,
    ).run(noteId);
    db.prepare("DELETE FROM q_history WHERE note_id = ?").run(noteId);
    db.prepare(
      `UPDATE note_q SET q_value = 0, update_count = 0, reward_sum = 0,
              reward_sq_sum = 0, last_reward = NULL
        WHERE note_id = ?`,
    ).run(noteId);
  })();
}

// --- Read ---

export function getQ(db: Database.Database, noteId: string): number {
  noteId = slugify(noteId);
  const row = db
    .prepare(
      `SELECT q_value, update_count, ${IS_LEGACY("q")} AS legacy
         FROM note_q q WHERE note_id = ?`,
    )
    .get(noteId) as { q_value: number; update_count: number; legacy: number } | undefined;
  // A row created by `incrementExposure` carries the column default, which is
  // 0.5 in every table created before #37. It is not a learned value, and
  // reading it as one made the first real update start from the old constant.
  if (!row || row.update_count === 0 || row.legacy) return DEFAULT_Q;
  return row.q_value;
}

export function getDecayedQ(db: Database.Database, noteId: string): number {
  noteId = slugify(noteId);
  const row = db
    .prepare(
      `SELECT q_value, update_count, last_updated, ${IS_LEGACY("q")} AS legacy
         FROM note_q q WHERE note_id = ?`,
    )
    .get(noteId) as
    | { q_value: number; update_count: number; last_updated: string; legacy: number }
    | undefined;

  // No row, or a row created by `incrementExposure` and never rewarded: there
  // is no learned value to decay. Decaying the initialisation constant made
  // `q_reranking` order notes by when exposure happened to create their row,
  // which is noise wearing a learned score's clothes. Legacy rows likewise
  // carry no usable value (see LEGACY_NOTES_SQL).
  if (!row || row.update_count === 0 || row.legacy) return DEFAULT_Q;

  return applyDecay(row.q_value, row.last_updated);
}

/** Q-informed time decay: high-Q notes decay slower, low-Q notes faster. */
function applyDecay(qValue: number, lastUpdated: string): number {
  const daysSince =
    (Date.now() - parseSqlTimestamp(lastUpdated)) / 86_400_000;

  // Tiers on the #37 scale (Q starts at 0 and rises with credit; ~12 straight
  // forward citations reach 0.7). The old 0.7 / 0.3 cut-offs were set for a
  // 0.5 start and put nearly every learned note in the fast tier.
  let mult = 1.0;
  if (qValue >= SLOW_DECAY_Q) mult = 0.7;
  else if (qValue <= FAST_DECAY_Q) mult = 1.3;

  return qValue * Math.exp(-DECAY_RATE * mult * daysSince);
}

/**
 * Milliseconds for a SQLite `datetime('now')` string, which is UTC and carries
 * no zone marker.
 *
 * `new Date("2026-09-15 16:22:39")` is parsed as LOCAL time, so west of UTC
 * every freshly written row looked like it was stamped in the future:
 * `daysSince` went negative and the decay became a small GROWTH, inflating
 * un-decayed values by the size of the UTC offset. Harmless at a 99-day
 * half-life, not harmless when the comparison being made is between two notes
 * that have learned nothing.
 */
function parseSqlTimestamp(value: string): number {
  const iso = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)
    ? `${value.replace(" ", "T")}Z`
    : value;
  return new Date(iso).getTime();
}

/**
 * A Q-value together with the provenance that says whether it means anything.
 *
 * `learned === false` means the value is the initialisation constant, not a
 * score: either the note has no row at all, or it has one written by
 * `incrementExposure` and never touched by a reward. Fix-list item 5 measured
 * 707 of 717 production rows in that state while every consumer read them
 * through `getQ` and could not tell them apart from a converged 0.5.
 */
export interface QState {
  noteId: string;
  q: number;
  decayedQ: number;
  updateCount: number;
  exposureCount: number;
  learned: boolean;
  /** Learned under the pre-#37 rules and ignored (see LEGACY_NOTES_SQL).
   *  `learned` is false for these; `updateCount` still reports the stored count. */
  legacy: boolean;
  lastUpdated: string | null;
}

/**
 * Full Q state for a note, in one query.
 *
 * Use this instead of `getQ`/`getDecayedQ` wherever the difference between
 * "no signal yet" and "learned, and it landed near the default" changes what
 * the caller should do - reranking weight, confidence reporting, health.
 */
export function getQState(db: Database.Database, noteId: string): QState {
  const id = slugify(noteId);
  const row = db
    .prepare(
      `SELECT q_value, update_count, exposure_count, last_updated,
              ${IS_LEGACY("q")} AS legacy
       FROM note_q q WHERE note_id = ?`,
    )
    .get(id) as
    | {
        q_value: number;
        update_count: number;
        exposure_count: number;
        last_updated: string;
        legacy: number;
      }
    | undefined;

  if (!row) {
    return {
      noteId: id,
      q: DEFAULT_Q,
      decayedQ: DEFAULT_Q,
      updateCount: 0,
      exposureCount: 0,
      learned: false,
      legacy: false,
      lastUpdated: null,
    };
  }

  const legacy = !!row.legacy;
  const learned = row.update_count > 0 && !legacy;
  return {
    noteId: id,
    // Unlearned rows carry the column default (0.5 on pre-#37 tables).
    q: learned ? row.q_value : DEFAULT_Q,
    // An un-updated row has no meaningful `last_updated` to decay from: it was
    // stamped when exposure created the row. Decaying it would manufacture a
    // difference between two notes that have both learned nothing.
    decayedQ: learned ? applyDecay(row.q_value, row.last_updated) : DEFAULT_Q,
    updateCount: row.update_count,
    exposureCount: row.exposure_count,
    learned,
    legacy,
    lastUpdated: row.last_updated,
  };
}

/**
 * Reward statistics for UCB, plus the two facts that say what they are worth.
 *
 * `learned` distinguishes "never rewarded" from "rewarded, converged near the
 * default" - see `getQState`. `exposure` rides along because the caller that
 * needs UCB also needs it (`explorationBonus` damps by it) and it is the same
 * row: one query, not two.
 */
export function getRewardStats(
  db: Database.Database,
  noteId: string,
): {
  mean: number;
  variance: number;
  count: number;
  exposure: number;
  learned: boolean;
} {
  noteId = slugify(noteId);
  const row = db
    .prepare(
      `SELECT update_count, reward_sum, reward_sq_sum, exposure_count,
              ${IS_LEGACY("q")} AS legacy
       FROM note_q q WHERE note_id = ?`,
    )
    .get(noteId) as
    | {
        update_count: number;
        reward_sum: number;
        reward_sq_sum: number;
        exposure_count: number;
        legacy: number;
      }
    | undefined;

  if (!row || row.update_count === 0 || row.legacy)
    return {
      mean: 0,
      variance: 0.25,
      count: 0,
      exposure: row?.exposure_count ?? 0,
      learned: false,
    };

  const mean = row.reward_sum / row.update_count;
  const variance = row.reward_sq_sum / row.update_count - mean * mean;
  return {
    mean,
    variance: Math.max(0, variance),
    count: row.update_count,
    exposure: row.exposure_count,
    learned: true,
  };
}

export function getExposureCount(
  db: Database.Database,
  noteId: string,
): number {
  noteId = slugify(noteId);
  const row = db
    .prepare("SELECT exposure_count FROM note_q WHERE note_id = ?")
    .get(noteId) as { exposure_count: number } | undefined;
  return row?.exposure_count ?? 0;
}

export function getTotalQUpdates(db: Database.Database): number {
  const row = db
    // Legacy updates do not count toward the lambda warm-up: they are not
    // learning the reranker should trust (see LEGACY_NOTES_SQL).
    .prepare(
      "SELECT COALESCE(SUM(update_count), 0) as total FROM note_q WHERE rule = 1",
    )
    .get() as { total: number };
  return row.total;
}

export function getTotalQueryCount(db: Database.Database): number {
  const row = db
    .prepare(
      "SELECT COUNT(DISTINCT session_id || '|' || query_text) as total FROM retrieval_log",
    )
    .get() as { total: number };
  return row.total;
}

// --- Write ---

/**
 * Where a Q-update came from. Recorded on every `q_history` row so a future
 * audit can separate deliberate session-end credit from anything else without
 * reverse-engineering the reward values (which is how the 2026-08 proxy
 * contamination had to be diagnosed: by matching rewards against the formula
 * `0.05/log2(rank+2)` after the fact).
 */
export type RewardSource =
  | "session_batch"
  | "explore_conclude"
  | "manual"
  | "migration";

/**
 * Sources permitted to write Q-values in normal operation.
 *
 * Deliberately narrow. Per-query writes are what produced the degenerate
 * feedback loop — a note rewarded for appearing in results the ranker itself
 * produced. Adding a source here is a decision about the learning signal, not
 * a plumbing detail: it belongs in review, which is the point of the guard.
 */
const ALLOWED_SOURCES: ReadonlySet<RewardSource> = new Set<RewardSource>([
  "session_batch",
  "explore_conclude",
]);

/**
 * Update a note's Q-value by EMA and record the transition.
 *
 * @param source  Provenance of this update. Anything outside ALLOWED_SOURCES
 *                throws unless `allowUnsafe` is set, so a future per-query
 *                write fails loudly at the first call in development instead
 *                of quietly accumulating for five months.
 * @param allowUnsafe  Escape hatch for migrations and one-off repair scripts.
 */
export function updateQ(
  db: Database.Database,
  noteId: string,
  reward: number,
  sessionId: string,
  source: RewardSource = "session_batch",
  allowUnsafe = false,
): void {
  if (!allowUnsafe && !ALLOWED_SOURCES.has(source)) {
    throw new Error(
      `updateQ: refusing write from source '${source}'. Q-values may only be ` +
        `written at session end (session_batch) or on explore conclusion ` +
        `(explore_conclude). Per-query writes create a degenerate feedback ` +
        `loop — see notes/the-ori-q-value-proxy-reward-was-a-degenerate-` +
        `feedback-loop. Pass allowUnsafe=true only from a migration script.`,
    );
  }

  noteId = slugify(noteId);
  if (!isKnownNote(db, noteId)) return;
  // First fixed-rule credit on a note with legacy learning: archive the legacy
  // history and start from 0, or this note would stay classified as legacy
  // (and ignored) forever, with reward sums mixing both rule sets.
  const legacy = db
    .prepare(`SELECT 1 FROM note_q q WHERE note_id = ? AND ${IS_LEGACY("q")}`)
    .get(noteId);
  if (legacy) retireLegacyNote(db, noteId);
  // From the decayed value (#37 review): starting from the stored value let one
  // small credit restore a note that had decayed for a year to its old peak.
  const oldQ = getDecayedQ(db, noteId);
  // Credits are uses, so a credit never lowers Q (#37 review). The EMA pulled a
  // note at 0.72 down to meet a 0.4 re-recall - a demotion for being used.
  // Positive rewards approach 1 in proportion to their strength; a negative
  // reward still moves toward it. The accumulator no longer produces negative
  // credits, and checkpoints written under the old rules are discarded on
  // recovery (serve.ts), but this function does not itself refuse them.
  const newQ =
    reward >= 0
      ? oldQ + ALPHA * reward * (1 - oldQ)
      : oldQ + ALPHA * (reward - oldQ);

  db.prepare(
    `
    INSERT INTO note_q (note_id, q_value, update_count, reward_sum, reward_sq_sum, last_updated, last_reward, rule)
    VALUES (?, ?, 1, ?, ?, datetime('now'), ?, 1)
    ON CONFLICT(note_id) DO UPDATE SET
      rule = 1,
      q_value = ?,
      update_count = update_count + 1,
      reward_sum = reward_sum + ?,
      reward_sq_sum = reward_sq_sum + ?,
      last_updated = datetime('now'),
      last_reward = ?
  `,
  ).run(
    noteId,
    newQ,
    reward,
    reward * reward,
    reward,
    newQ,
    reward,
    reward * reward,
    reward,
  );

  db.prepare(
    `
    INSERT INTO q_history (note_id, old_q, new_q, reward, reward_source, session_id)
    VALUES (?, ?, ?, ?, ?, ?)
  `,
  ).run(noteId, oldQ, newQ, reward, source, sessionId);
}

/**
 * True unless the derived index exists, is populated, and has no note with
 * this slug (#41). Dangling link targets and parser false positives must not
 * get learner state. With no index (degraded mode) nothing is refused.
 */
export function isKnownNote(db: Database.Database, slug: string): boolean {
  try {
    if (!db.prepare("SELECT 1 FROM note LIMIT 1").get()) return true;
    // note.slug holds the file basename; compare in slug space. The exact
    // match is the common case; only a miss pays for the full scan.
    if (db.prepare("SELECT 1 FROM note WHERE slug = ?").get(slug)) return true;
    return indexedSlugs(db).has(slug);
  } catch {
    return true; // no note table
  }
}

function indexedSlugs(db: Database.Database): Set<string> {
  const rows = db.prepare("SELECT slug FROM note").all() as { slug: string }[];
  return new Set(rows.map((r) => slugify(r.slug)));
}

/**
 * Delete never-credited note_q rows whose note does not exist (#41). They
 * hold no earned reward by definition, only exposure. Rows with credit are
 * kept: a renamed or deleted note's history is not this function's call.
 */
export function pruneUnknownQRows(db: Database.Database): number {
  try {
    if (!db.prepare("SELECT 1 FROM note LIMIT 1").get()) return 0;
    const known = indexedSlugs(db);
    const rows = db
      .prepare("SELECT note_id FROM note_q WHERE update_count = 0")
      .all() as { note_id: string }[];
    const del = db.prepare("DELETE FROM note_q WHERE note_id = ?");
    let n = 0;
    for (const r of rows) {
      if (!known.has(r.note_id)) n += del.run(r.note_id).changes;
    }
    return n;
  } catch {
    return 0; // no note or note_q table
  }
}

export function incrementExposure(
  db: Database.Database,
  noteId: string,
): void {
  noteId = slugify(noteId);
  if (!isKnownNote(db, noteId)) return;
  db.prepare(
    `
    INSERT INTO note_q (note_id, exposure_count)
    VALUES (?, 1)
    ON CONFLICT(note_id) DO UPDATE SET exposure_count = exposure_count + 1
  `,
  ).run(noteId);
}

export function logRetrieval(
  db: Database.Database,
  sessionId: string,
  queryText: string,
  queryType: string,
  noteId: string,
  rank: number,
  simScore: number,
  qScore: number,
  ucbBonus: number,
  finalScore: number,
): void {
  noteId = slugify(noteId);
  db.prepare(
    `
    INSERT INTO retrieval_log
      (session_id, query_text, query_type, note_id, rank,
       similarity_score, q_score, ucb_bonus, final_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `,
  ).run(
    sessionId,
    queryText,
    queryType,
    noteId,
    rank,
    simScore,
    qScore,
    ucbBonus,
    finalScore,
  );
}

// --- Exploration ---

/**
 * Exposure damping factor for the exploration bonus, in [MIN_EXPLORE_RETENTION, 1].
 *
 * Monotonically decreasing in exposure and floored, never zero. A note nobody
 * has seen keeps its whole bonus; one shown 100 times keeps a fifth of it.
 */
export function exposureDamping(exposure: number): number {
  if (exposure <= 0) return 1;
  return Math.max(
    Math.pow(1 + exposure, -EXPLORE_EXPOSURE_BETA),
    MIN_EXPLORE_RETENTION,
  );
}

/**
 * Exploration bonus, damped by how often a never-credited note was already shown.
 *
 * ## Why exposure enters here (fix list item 8, 2026-09-15)
 *
 * Measured on the live vault: the top 50 notes held 47.2% of all exposure and
 * ~280 of 1,423 notes had never been surfaced once. The cause is visible in
 * the old one-line form of this function. With 707 of 717 rows at
 * `update_count = 0`, `count === 0` held for nearly every candidate, so nearly
 * every candidate received exactly the same `c * 2.5`. A constant added to
 * every score is not exploration - it cancels in the ranking, leaving
 * similarity alone to decide, and similarity is what concentrated exposure in
 * the first place.
 *
 * Damping restores the differential that constant destroyed. It is
 * deterministic and monotone in exposure, and there is no early return above
 * it that could preempt it: the 2026-09-12 stage starvation bug was precisely
 * a short-circuit evaluated before the mechanism that guarantees recovery.
 *
 * `stats.exposure` is optional. Omitted means "no exposure information" and
 * damps nothing.
 *
 * ## Exposure damps only notes never credited (#37 follow-up)
 *
 * This used to be UCB-Tuned: `c * 2.5` at zero credits, then a variance term
 * shrinking with `count` from the first credit on. The first credit cut the
 * bonus roughly in half while adding only ALPHA * reward of Q, so a note that
 * was used ranked BELOW an equally-shown note that was not - the #37
 * inversion again, through this term instead of DEFAULT_Q.
 *
 * Two intermediate fixes failed review for the same reason one step removed:
 * damping every exposure, then damping `exposure - count`. Exposure is counted
 * per query and credit per session (at most one per note), so normal use -
 * shown in two queries, re-recalled once - still left "unused" exposure that
 * one credit could not repay, and a used note lost to a never-shown equal
 * match (Codex review, reproduced with the real accumulator).
 *
 * The bonus exists to rescue notes that keep being shown and ignored. Once a
 * note has earned a credit, its exposures are not evidence against it, so it
 * keeps the full bonus - the same as an unseen note - and Q decides between
 * them. A credit therefore never lowers the bonus, and a used note always
 * outscores an equally relevant unseen one. Uncredited notes, including legacy
 * rows (count 0), are damped by exposure as fix-list item 8 requires.
 * `mean`, `variance` and `totalQueries` are ignored.
 */
export function explorationBonus(
  stats: {
    mean: number;
    variance: number;
    count: number;
    exposure?: number;
  },
  _totalQueries: number,
  c: number = 0.2,
): number {
  if (stats.count > 0) return c * 2.5;
  return c * 2.5 * exposureDamping(stats.exposure ?? 0);
}

/**
 * Probability that one returned slot is handed to a never-surfaced note.
 *
 * The exploration bonus alone cannot fix exposure bias, and the arithmetic
 * says why: `phaseB` blends z-scored similarity at weight (1 - lambda) >= 0.5,
 * and a z-normalized candidate list spans roughly three standard deviations,
 * so the largest bonus this module can emit (c * 2.5 = 0.5) moves a note about
 * two ranks. A note that never enters the window cannot be promoted out of it.
 * Measured consequence: ~280 of 1,423 notes had never been surfaced once while
 * the top 50 held 47.2% of all exposure.
 *
 * 0.1 gives a cold note a slot in one query out of ten - the same escape-hatch
 * argument as the stage bandit's EPSILON, at the same order of magnitude
 * (docs/stage-bandit-starvation.md), and it costs one of eight returned slots
 * when it fires.
 */
export const COLD_START_EPSILON = 0.1;

export interface ColdStartOptions {
  epsilon?: number;
  /** Injectable for tests; a stochastic path with an unpinned RNG is a flake. */
  random?: () => number;
}

/**
 * Hand one of the top-`k` slots to the best never-surfaced candidate, epsilon
 * of the time.
 *
 * "Never surfaced" means no `note_q` row or `exposure_count = 0` - the note has
 * never been shown to an agent, so nothing about it has ever been learned and
 * ranking it on its Q-value ranks the initialisation constant.
 *
 * The epsilon draw happens FIRST, before any early exit. That ordering is the
 * whole lesson of the 2026-09-12 stage starvation bug, where a budget
 * short-circuit sat above the epsilon check and six stages stayed dark for six
 * days: the mechanism that guarantees recovery must not be reachable only when
 * some other condition happens to allow it. Here it also keeps RNG consumption
 * independent of the candidate list, so a caller's random stream does not
 * change shape with vault size.
 *
 * Returns the top-`k` slice, with at most one substitution. Never grows the
 * list and never reorders anything else.
 */
export function applyColdStartFloor<T extends { title: string }>(
  db: Database.Database,
  ranked: T[],
  k: number,
  opts: ColdStartOptions = {},
): T[] {
  const { epsilon = COLD_START_EPSILON, random = Math.random } = opts;
  const fires = random() < epsilon;

  if (k <= 0) return [];
  const top = ranked.slice(0, k);
  if (!fires || ranked.length <= k) return top;

  // Candidates that did not make the cut, in rank order: the first cold one is
  // the strongest note nobody has seen.
  const below = ranked.slice(k);
  const ids = below.map((c) => slugify(c.title));
  const placeholders = ids.map(() => "?").join(",");
  const surfaced = new Set(
    (
      db
        .prepare(
          `SELECT note_id FROM note_q
           WHERE exposure_count > 0 AND note_id IN (${placeholders})`,
        )
        .all(...ids) as { note_id: string }[]
    ).map((r) => r.note_id),
  );

  const coldIndex = ids.findIndex((id) => !surfaced.has(id));
  if (coldIndex < 0) return top;

  // Costs the weakest kept slot, never the head of the list.
  top[k - 1] = below[coldIndex]!;
  return top;
}

// --- Batch update ---

/**
 * Apply a session's worth of credit in one transaction.
 *
 * This is the sanctioned write path. `source` defaults to session_batch and is
 * forwarded to `updateQ`, which enforces ALLOWED_SOURCES — so a caller cannot
 * launder a per-query write through the batch helper.
 */
export function batchUpdateQ(
  db: Database.Database,
  rewards: Map<string, number>,
  sessionId: string,
  source: RewardSource = "session_batch",
): void {
  const tx = db.transaction(() => {
    for (const [noteId, reward] of rewards) {
      updateQ(db, noteId, reward, sessionId, source);
    }
  });
  tx();
}

/**
 * Health snapshot of the learning signal, for `ori_health` and for tests.
 *
 * The 2026-08 failure was invisible for five months because nothing summarized
 * *what kind* of reward was accumulating. These numbers would have made it
 * obvious within a week:
 *
 *   - `bySource` — a per-query source dominating session_batch is the alarm.
 *   - `forwardCitations` — 0 over many sessions means key matching is broken.
 *   - `exposureQCorrelation` — should be >= 0. Negative means the system is
 *     punishing use, which is the degenerate-loop signature.
 *   - `distinctKeyShapes` — >1 means slug/title drift has returned.
 *
 * The last three were added 2026-09-15 for fix-list item 5, which was invisible
 * for the opposite reason: nothing counted the rows where learning had *not*
 * happened. Production held 717 tracked notes, 707 of them never updated, and
 * every summary in the system reported only the 10 that were.
 *
 *   - `neverUpdated` — rows sitting at the initialisation constant.
 *   - `exposedButNeverUpdated` — shown to an agent, never credited. A large
 *     value means retrieval is running and the session flush is not.
 *   - `neverExposed` — tracked but never surfaced; the exposure-bias tail.
 */
export function getLearningHealth(db: Database.Database): {
  bySource: Record<string, number>;
  forwardCitations: number;
  exposureQCorrelation: number;
  distinctKeyShapes: number;
  totalUpdates: number;
  trackedNotes: number;
  neverUpdated: number;
  exposedButNeverUpdated: number;
  neverExposed: number;
  /** Notes whose pre-#37 learning is ignored at read time (LEGACY_NOTES_SQL). */
  legacyNotes: number;
} {
  const bySource: Record<string, number> = {};
  const sourceRows = db
    .prepare("SELECT reward_source, COUNT(*) as n FROM q_history GROUP BY reward_source")
    .all() as { reward_source: string; n: number }[];
  for (const r of sourceRows) bySource[r.reward_source] = r.n;

  // A +1.0 reward is only ever a forward citation (reward.ts), and since #37
  // no exposure divisor scales it, so it is written as exactly 1.0. Rounding
  // guards against float drift. Keep in sync with CREDITED_SIGNALS.
  const fc = db
    // Legacy notes excluded, because health.ts compares this to totalUpdates:
    // an ignored legacy citation must not mask zero fresh ones. Approximate
    // after import-learned, which appends history: a note restored as
    // fixed-rule can carry older rows here. A diagnostic, not a ranking input.
    .prepare(
      `SELECT COUNT(*) as n FROM q_history
        WHERE ROUND(reward, 6) = 1.0 AND note_id NOT IN (${LEGACY_NOTES_SQL})`,
    )
    .get() as { n: number };

  // Pearson correlation between exposure and learned value. Computed in SQL to
  // avoid pulling the whole table into memory on large vaults.
  // Legacy rows are excluded: they are ignored by ranking, and their built-in
  // negative correlation (retrieved more -> pulled toward 0 more) would keep
  // this warning firing forever on any vault that declined the reset.
  const stats = db
    .prepare(
      `SELECT COUNT(*) n, SUM(exposure_count) sx, SUM(q_value) sy,
              SUM(exposure_count * q_value) sxy,
              SUM(exposure_count * exposure_count) sxx,
              SUM(q_value * q_value) syy
       FROM note_q q WHERE update_count > 0 AND exposure_count > 0
         AND NOT ${IS_LEGACY("q")}`,
    )
    .get() as Record<string, number>;
  let corr = 0;
  if (stats.n > 1) {
    const num = stats.n * stats.sxy - stats.sx * stats.sy;
    const den = Math.sqrt(
      (stats.n * stats.sxx - stats.sx * stats.sx) *
        (stats.n * stats.syy - stats.sy * stats.sy),
    );
    corr = den === 0 ? 0 : num / den;
  }

  // Key shapes: slugs contain no spaces and no uppercase. Anything else means
  // a raw title leaked past slugify().
  const shapes = db
    .prepare(
      `SELECT COUNT(DISTINCT CASE WHEN note_id LIKE '% %' THEN 'title' ELSE 'slug' END) as n
       FROM note_q`,
    )
    .get() as { n: number };

  // Signal-absence counters. One pass, so this stays cheap on large vaults.
  const coverage = db
    .prepare(
      `SELECT COUNT(*) total,
              SUM(CASE WHEN update_count = 0 THEN 1 ELSE 0 END) never_updated,
              SUM(CASE WHEN update_count = 0 AND exposure_count > 0 THEN 1 ELSE 0 END) exposed_unlearned,
              SUM(CASE WHEN exposure_count = 0 THEN 1 ELSE 0 END) never_exposed
       FROM note_q`,
    )
    .get() as Record<string, number>;

  return {
    bySource,
    forwardCitations: fc.n,
    exposureQCorrelation: corr,
    distinctKeyShapes: shapes.n,
    totalUpdates: getTotalQUpdates(db),
    trackedNotes: coverage.total ?? 0,
    neverUpdated: coverage.never_updated ?? 0,
    exposedButNeverUpdated: coverage.exposed_unlearned ?? 0,
    neverExposed: coverage.never_exposed ?? 0,
    legacyNotes: (
      db.prepare(`SELECT COUNT(*) AS n FROM (${LEGACY_NOTES_SQL})`).get() as { n: number }
    ).n,
  };
}

// Re-export constants for tests
export {
  ALPHA,
  DEFAULT_Q,
  DECAY_RATE,
  EXPOSURE_BETA,
  ALLOWED_SOURCES,
  EXPLORE_EXPOSURE_BETA,
  MIN_EXPLORE_RETENTION,
};
