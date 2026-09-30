/**
 * Phase B Q-value reranking.
 * Takes the top-k1 candidates from RRF fusion (Phase A) and reranks
 * them using a lambda blend of similarity score and learned Q-value,
 * plus an exposure-damped exploration bonus, with cumulative bias cap.
 *
 * Research: MemRL two-phase, Drift invariants, CIKM 2024 exposure bias
 */

import type Database from "better-sqlite3";
import type { ScoredNote } from "./ranking.js";
import {
  getDecayedQ,
  getRewardStats,
  getTotalQUpdates,
  explorationBonus,
  incrementExposure,
  logRetrieval,
  applyColdStartFloor,
  type ColdStartOptions,
} from "./qvalue.js";

// Constants
/**
 * Was 0.15. Raised with the #37 follow-up: every vault restarts the warm-up
 * after upgrading (legacy updates no longer count), and at 0.15 the largest Q
 * contribution (0.15 / Q_SCALE = 0.30) was below the largest exploration gap
 * (0.425), so for the first 200 credits no amount of use could lift a note over
 * an equally relevant never-shown one - #37 again, for months, right after the
 * fix shipped (Codex review, reproduced). 0.3 keeps the cap (0.60) above the
 * gap from the first query. bench/lambda-sweep.mjs found recall@5 flat from
 * 0 to 0.35, but it replayed the OLD blend (z-scored Q, no exploration term),
 * so that range is indicative, not a validation of this formula. The
 * 1,566-note replay behind this change measured top-10 overlap 96% and the
 * same #1 in 39/40 queries against Q reranking disabled.
 */
const LAMBDA_MIN = 0.3;
// Measured, not chosen. bench/lambda-sweep.mjs replays 1,653 real query
// instances from retrieval_log at every lambda in [0, 0.6] -- both blend
// inputs are logged per candidate, so the re-ranking is exact rather than
// simulated. Term-coverage recall@5 is flat from 0 to 0.35 and then declines
// monotonically. Paired bootstrap over the same instances, 5000 resamples:
//
//     0.35  0.5618            best
//     0.40  -0.0048  CI [-0.0067, -0.0037]   significant
//     0.55  -0.0179  CI [-0.0254, -0.0149]   significant
//     0.60  -0.0262  CI [-0.0332, -0.0210]   significant
//
// A QUERY_TYPE_SHIFTS table used to sit here adding -0.10 semantic, +0.15
// procedural, +0.05 decision, 0 episodic. Every entry moved lambda away from
// the optimum and the two positive shifts landed on the two worst points
// measured. It also mis-stated itself: base was already LAMBDA_MAX at
// maturity, so procedural's +0.15 hit the hard 0.6 clamp and delivered +0.10.
// And it could not have been earning its keep either way -- query_type is
// "semantic" on 15,463 of 15,892 logged rows, a 0.0047 traffic-weighted
// deviation from a constant.
//
// Intent still drives the type-space vector (engine.ts buildQueryTypeVec) and
// the space/split weight profiles (intent.ts). It no longer moves lambda.
const LAMBDA_MAX = 0.35;
const LAMBDA_MATURITY = 200;
const MAX_CUMULATIVE_BIAS = 3.0;
const EXCESS_COMPRESSION = 0.3;
const K2 = 8;
/**
 * Divisor that puts Q on the blend's scale (#37 review).
 *
 * Constraint: the largest Q contribution, LAMBDA_MAX / Q_SCALE, must exceed the
 * largest exploration gap between two candidates, c * 2.5 * (1 -
 * MIN_EXPLORE_RETENTION) = 0.425 (qvalue.ts explorationBonus). Since credited
 * notes are no longer exposure-damped, a used note already matches an unseen
 * one on the bonus and wins on Q; the constraint still governs a used note
 * against an uncredited note that has been shown less. At 0.5 the cap is 0.70.
 * 2.0 was tried and let exploration outweigh any amount of use.
 */
const Q_SCALE = 0.5;

// --- Z-score normalization ---

export function zNormalize(values: number[]): number[] {
  const n = values.length;
  if (n === 0) return [];
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const std =
    Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return values.map((v) => (v - mean) / std);
}

// --- Lambda ---

export function computeLambda(totalQUpdates: number): number {
  // Warm-up ramp only: trust similarity until enough Q-updates exist to mean
  // anything, then hold at LAMBDA_MAX. Clamping t rather than the result keeps
  // the output inside [LAMBDA_MIN, LAMBDA_MAX] by construction, so no later
  // clamp can silently truncate a configured value the way the old 0.6 bound
  // truncated procedural's +0.15 down to +0.10.
  const t = Math.min(Math.max(totalQUpdates / LAMBDA_MATURITY, 0), 1);
  return LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * t;
}

// --- Phase B ---

export function phaseB(
  db: Database.Database,
  candidates: ScoredNote[],
  queryText: string,
  queryType: string,
  sessionId: string,
  coldStart: ColdStartOptions = {},
  /**
   * Record exposure + retrieval_log for the top K. The ranked-query pipeline
   * passes false because it records exactly what it RETURNS after archive
   * filtering and trimming; recording here too counted every returned note
   * twice (#37 review), doubling exposure-driven effects.
   */
  record = true,
  /** Filled with each returned note's blend inputs, so a caller that records
   *  retrieval_log itself logs real values rather than the final score. */
  debugOut?: Map<string, { simNorm: number; qNorm: number; ucb: number }>,
  /**
   * `k`: slots the cold-start floor competes for (default K2).
   * `keepTail`: return every candidate, reranked, instead of only the top k.
   * The ranked-query pipeline filters archived notes and trims to the
   * caller's limit AFTER this; truncating here to K2 = 8 capped every
   * request at 8 results and let archived notes consume those 8 slots.
   */
  opts: { k?: number; keepTail?: boolean } = {},
): ScoredNote[] {
  if (candidates.length === 0) return [];

  const totalUpdates = getTotalQUpdates(db);
  const lambda = computeLambda(totalUpdates);

  // Get raw scores
  const simRaw = candidates.map((c) => c.score);
  const qRaw = candidates.map((c) => getDecayedQ(db, c.title));

  // Z-score normalize both (CRITICAL — without this lambda is meaningless)
  const simNorm = zNormalize(simRaw);
  // Q is NOT z-scored (#37 review). It already lives on a fixed scale, and
  // with most candidates unlearned (Q = 0) z-scoring gave one credited note
  // +sqrt(n-1) (6.24 at n = 40) whatever its value - a Q of 0.017 and 0.9
  // got the same boost. Centre on the candidate mean, divide by a constant:
  // Q = 1.0 above an all-zero field adds lambda * 2 to the blend (0.70 at
  // maturity). See Q_SCALE.
  const qMean = qRaw.reduce((a, b) => a + b, 0) / (qRaw.length || 1);
  const qNorm = qRaw.map((q) => (q - qMean) / Q_SCALE);

  const results = candidates.map((c, i) => {
    // Lambda blend
    const blended = (1 - lambda) * simNorm[i] + lambda * qNorm[i];

    // Exploration bonus (exposure-damped for never-credited notes)
    const stats = getRewardStats(db, c.title);
    // totalQueries is unused by explorationBonus since the #37 follow-up;
    // counting it scanned all of retrieval_log on every query (~5 ms at 15k rows).
    const ucb = explorationBonus(stats, 0);

    // Raw Phase B score
    let score = blended + ucb;

    // Cumulative bias cap (Drift invariant — prevents runaway boosts)
    const maxAllowed = c.score * MAX_CUMULATIVE_BIAS;
    if (score > maxAllowed) {
      score = maxAllowed + (score - maxAllowed) * EXCESS_COMPRESSION;
    }

    return {
      ...c,
      score,
      _phaseB: { simNorm: simNorm[i], qNorm: qNorm[i], ucb, lambda },
    };
  });

  // Sort, then let the cold-start floor claim one slot before the cut. It must
  // run on the FULL ranked list: applied after `slice`, every never-surfaced
  // note has already been discarded and the floor can only reorder notes that
  // were going to be returned anyway. This is the same ordering lesson as
  // docs/stage-bandit-starvation.md - the recovery mechanism goes above the
  // cutoff, not below it.
  results.sort((a, b) => b.score - a.score);
  const topK = applyColdStartFloor(db, results, opts.k ?? K2, coldStart);
  const out = opts.keepTail
    ? [...topK, ...results.filter((r) => !topK.includes(r))]
    : topK;

  // Exposure counts what an agent was actually shown, not what was considered.
  // Before 2026-09-15 this incremented for every candidate, which made
  // `exposure_count = 0` unreachable for anything that ever reached phaseB and
  // left the cold-start floor with nothing to find. It also overstated the
  // exposure divisor in reward.ts for notes that were never returned.
  for (let rank = 0; record && rank < topK.length; rank++) {
    const r = topK[rank];
    incrementExposure(db, r.title);
    logRetrieval(
      db,
      sessionId,
      queryText,
      queryType,
      r.title,
      rank,
      r._phaseB.simNorm,
      r._phaseB.qNorm,
      r._phaseB.ucb,
      r.score,
    );
  }

  if (debugOut) {
    for (const r of out) {
      debugOut.set(r.title, { simNorm: r._phaseB.simNorm, qNorm: r._phaseB.qNorm, ucb: r._phaseB.ucb });
    }
  }

  // Strip internal debug data before returning
  return out.map(({ _phaseB, ...rest }) => rest) as ScoredNote[];
}

// Re-export constants for tests
export {
  LAMBDA_MIN,
  LAMBDA_MAX,
  LAMBDA_MATURITY,
  MAX_CUMULATIVE_BIAS,
  EXCESS_COMPRESSION,
  K2,
};
