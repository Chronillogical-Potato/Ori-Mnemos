/**
 * Opt-in reset of Q-values learned under the pre-#37 reward rules.
 *
 * Before #37, `DEFAULT_Q` was 0.5 and every typical reward sat below it, so
 * the EMA lowered a note's Q each time it was used. Every vault that learned
 * under those rules holds Q-values that rank used notes below unused ones.
 * The fixed rules stop new damage but do not repair stored values.
 *
 * The reset is offered, not forced: the MCP server surfaces a notice to the
 * agent, the agent asks the user, and `ori_learning_reset` records the answer.
 * The same ask-until-answered shape as the update notice (update-check.ts).
 *
 * What a reset does, and deliberately does not do:
 *   - clears learned state in `note_q` (q_value, update_count, reward sums,
 *     last_reward) back to "never learned";
 *   - archives `q_history` into `q_history_pre_37` rather than deleting it;
 *   - keeps `exposure_count`, `retrieval_log` and `co_occurrence`: those are
 *     observations of what was shown, not values learned from a bad rule;
 *   - backs the database file up first.
 */
import type Database from "better-sqlite3";

/** `meta` key holding the user's answer: "accepted:<iso>" or "declined:<iso>". */
export const LEARNING_RESET_KEY = "learning_reset_37";

/** Updates below this are too few to be worth asking about. */
export const MIN_UPDATES_TO_OFFER = 20;

/** The question is raised at most this many times, then goes quiet for good. */
export const MAX_RESET_REMINDERS = 3;

/** `meta` key counting how many sessions the question has been raised in. */
const SHOWN_KEY = "learning_reset_37_shown";

export interface LearningResetStatus {
  /** Q-updates written before the #37 rules took effect. */
  legacyUpdates: number;
  /** Notes whose Q was learned under those rules. */
  legacyNotes: number;
  /** Prior answer, if any. "declined" is permanent. */
  decision: "accepted" | "declined" | null;
  /** Sessions the question has already been raised in. */
  timesShown: number;
  /** True when the agent should raise the question this session. */
  offer: boolean;
}

function readDecision(db: Database.Database): "accepted" | "declined" | null {
  try {
    const row = db.prepare("SELECT value FROM meta WHERE key = ?").get(LEARNING_RESET_KEY) as
      | { value: string }
      | undefined;
    if (!row) return null;
    if (row.value.startsWith("accepted")) return "accepted";
    if (row.value.startsWith("declined")) return "declined";
    return null;
  } catch {
    return null;
  }
}

function readShown(db: Database.Database): number {
  try {
    const row = db.prepare("SELECT value FROM meta WHERE key = ?").get(SHOWN_KEY) as
      | { value: string }
      | undefined;
    return row ? Number(row.value) || 0 : 0;
  } catch {
    return 0;
  }
}

/**
 * Count one raising of the question. Called by the server only when a notice
 * is actually attached to a response, so the cap counts real asks, not
 * sessions that never reached a carrying tool.
 */
export function recordLearningResetShown(db: Database.Database): number {
  db.exec("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)");
  // One atomic statement: two servers on one vault cannot both read n and write n+1.
  const row = db
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, '1')
       ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
       RETURNING CAST(value AS INTEGER) AS n`,
    )
    .get(SHOWN_KEY) as { n: number };
  return row.n;
}

function writeDecision(db: Database.Database, decision: "accepted" | "declined"): void {
  db.exec("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)");
  db.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)").run(
    LEARNING_RESET_KEY,
    `${decision}:${new Date().toISOString()}`,
  );
}

/**
 * Whether this vault holds learning from the old rules, and whether to ask.
 *
 * "Legacy" is identified from the history itself, because the old code wrote no
 * version stamp: under the old rules a note's first update started from
 * `old_q = 0.5`, under the new ones from 0. Any note with such a row learned
 * under the old rules, and all of its history is counted. After an accepted
 * reset `q_history` is archived and empty, so the count drops to 0; after a
 * decline the stored answer stops the question.
 */
export function getLearningResetStatus(db: Database.Database): LearningResetStatus {
  const decision = readDecision(db);
  const timesShown = readShown(db);
  let legacyUpdates = 0;
  let legacyNotes = 0;
  // Answered or capped: nothing to ask, so skip the full scan of q_history,
  // which otherwise ran on every ranked query forever after a "no".
  if (decision !== null || timesShown >= MAX_RESET_REMINDERS) {
    return { legacyUpdates, legacyNotes, decision, timesShown, offer: false };
  }
  try {
    const row = db
      .prepare(
        `SELECT COUNT(*) AS n, COUNT(DISTINCT note_id) AS notes
           FROM q_history
          WHERE note_id IN (SELECT note_id FROM q_history WHERE old_q = 0.5)`,
      )
      .get() as { n: number; notes: number };
    legacyUpdates = row.n;
    legacyNotes = row.notes;
  } catch {
    // No learning tables yet: nothing to reset.
  }
  return {
    legacyUpdates,
    legacyNotes,
    decision,
    timesShown,
    offer:
      decision === null &&
      legacyUpdates >= MIN_UPDATES_TO_OFFER &&
      timesShown < MAX_RESET_REMINDERS,
  };
}

/**
 * The instruction the agent receives. Ask-shaped on purpose: inert metadata is
 * ignored, a question to relay gets relayed. Returns null when there is
 * nothing to ask.
 */
export function buildLearningResetNotice(status: LearningResetStatus): string | null {
  if (!status.offer) return null;
  const n = status.timesShown + 1;
  const last = n >= MAX_RESET_REMINDERS;
  return (
    `Learning reset question, reminder ${n} of ${MAX_RESET_REMINDERS}. ` +
    `Why: Ori found ${status.legacyUpdates} learned ranking updates on ${status.legacyNotes} notes computed ` +
    `under a scoring rule with a known defect (issue #37). Every note started at a score above almost any ` +
    `reward it could earn, so each time a note was used its score went down: notes the user actually relies ` +
    `on ended up ranked below notes that were never opened. The rule is fixed for new learning, but these ` +
    `stored scores are on the old scale and keep distorting search: the most-used notes still rank lowest ` +
    `among them, and they drown out anything learned under the fixed rule. ` +
    `Ask the user at a natural pause, in your own words: "Ori had a bug where notes you use a lot got ranked ` +
    `lower than notes you've never opened, because each use counted against them. It's fixed now, but the old ` +
    `scores are still skewing search. Want me to reset them so learning starts clean? Your notes aren't ` +
    `touched and the database is backed up first. You can say yes, not now, or no and never ask again` +
    (last ? `. This is the last time I'll ask."` : ` (I'll ask at most ${MAX_RESET_REMINDERS} times)."`) +
    ` If yes: call ori_learning_reset with decision="accepted". ` +
    `If no or never: call ori_learning_reset with decision="declined"; that is permanent. ` +
    `If not now, or no answer: do NOT call the tool` +
    (last ? `; this was the final reminder, so it will not be asked again.` : `; it will be asked again next session.`)
  );
}

/**
 * Back up, archive and clear old-rule learning. No guards and no decision
 * write: callers decide whether a reset is warranted.
 */
function resetLegacyLearning(
  db: Database.Database,
  dbPath: string,
): { backup: string; notesCleared: number; historyArchived: number } {
  // VACUUM INTO writes a transactionally consistent copy through SQLite
  // itself, WAL contents included. Copying the main file was not: with another
  // reader holding a snapshot the checkpoint could not complete and the copy
  // came out malformed.
  const backup = `${dbPath}.bak-learning-reset-37-${Date.now()}`;
  db.prepare("VACUUM INTO ?").run(backup);

  let notesCleared = 0;
  let historyArchived = 0;
  const tx = db.transaction(() => {
    db.exec(`
      CREATE TABLE IF NOT EXISTS q_history_pre_37 (
        id INTEGER, note_id TEXT, old_q REAL, new_q REAL,
        reward REAL, reward_source TEXT, session_id TEXT, timestamp TEXT
      )
    `);
    // Only notes that learned under the old rule. Learning made under the fixed
    // rule between the upgrade and a late "yes" is kept.
    const LEGACY = "SELECT note_id FROM q_history WHERE old_q = 0.5";
    historyArchived = db
      .prepare(
        `INSERT INTO q_history_pre_37
           SELECT id, note_id, old_q, new_q, reward, reward_source, session_id, timestamp
             FROM q_history WHERE note_id IN (${LEGACY})`,
      )
      .run().changes;
    notesCleared = db
      .prepare(
        `UPDATE note_q
            SET q_value = 0, update_count = 0, reward_sum = 0,
                reward_sq_sum = 0, last_reward = NULL
          WHERE note_id IN (${LEGACY})`,
      )
      .run().changes;
    db.exec(`DELETE FROM q_history WHERE note_id IN (${LEGACY})`);
  });
  tx();

  return { backup, notesCleared, historyArchived };
}

export interface LearningResetResult {
  decision: "accepted" | "declined";
  backup: string | null;
  notesCleared: number;
  historyArchived: number;
  /** Set when "accepted" was a no-op, with the reason. */
  skipped?: string;
}

function legacyCount(db: Database.Database): number {
  try {
    return (
      db.prepare("SELECT COUNT(*) AS n FROM q_history WHERE old_q = 0.5").get() as { n: number }
    ).n;
  } catch {
    return 0;
  }
}

/**
 * Record the user's answer, and on "accepted" perform the reset.
 *
 * `dbPath` is the file behind `db`, used only for the backup copy. The copy is
 * taken after a WAL checkpoint so it contains every committed write.
 */
export function applyLearningResetDecision(
  db: Database.Database,
  dbPath: string,
  decision: "accepted" | "declined",
): LearningResetResult {
  if (decision === "declined") {
    writeDecision(db, "declined");
    return { decision, backup: null, notesCleared: 0, historyArchived: 0 };
  }

  // Guard on the server side, not only in the tool description: a repeated or
  // unprompted "accepted" must not wipe learning made under the fixed rules.
  if (readDecision(db) === "accepted") {
    return { decision, backup: null, notesCleared: 0, historyArchived: 0, skipped: "already reset" };
  }
  if (readShown(db) === 0) {
    // Never offered (below the threshold, or the agent acted on its own).
    return { decision, backup: null, notesCleared: 0, historyArchived: 0, skipped: "not offered" };
  }
  const found = legacyCount(db);
  if (found === 0) {
    writeDecision(db, "accepted");
    return { decision, backup: null, notesCleared: 0, historyArchived: 0, skipped: "no old-rule learning" };
  }

  const done = resetLegacyLearning(db, dbPath);
  writeDecision(db, "accepted");
  return { decision, ...done };

}

/** Old-rule rows currently in q_history. Call before and after an import. */
export function countLegacyLearning(db: Database.Database): number {
  return legacyCount(db);
}

/**
 * After `import-learned`: if the import brought old-rule learning back into a
 * vault whose user already said yes, clear it again (with a fresh backup).
 * The user has answered; asking again would reset the 3-ask cap, and a "no"
 * on the re-ask would keep the bad scores for good. A declined or unanswered
 * vault is left alone - its answer (or the pending question) still stands.
 * Returns the old-rule rows the import added and whether they were cleared.
 */
export function reapplyLearningResetAfterImport(
  db: Database.Database,
  dbPath: string,
  legacyBefore: number,
): { added: number; cleared: boolean; backup: string | null } {
  const added = legacyCount(db) - legacyBefore;
  if (added > 0 && readDecision(db) === "accepted") {
    const r = resetLegacyLearning(db, dbPath);
    return { added, cleared: true, backup: r.backup };
  }
  return { added: Math.max(added, 0), cleared: false, backup: null };
}
