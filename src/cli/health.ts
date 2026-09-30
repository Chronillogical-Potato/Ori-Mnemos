import path from "node:path";
import { promises as fs } from "node:fs";
import { findVaultRoot, getVaultPaths, listNoteTitles } from "../core/vault.js";
import {
  buildGraph,
  findDanglingLinks,
  findOrphans,
  type LinkGraph,
} from "../core/graph.js";
import { loadConfig, resolveTemplatePath } from "../core/config.js";
import { validateNoteAgainstSchema } from "../core/schema.js";
import { parseFrontmatter } from "../core/frontmatter.js";
import { computeVitality } from "../core/vitality.js";
import { initDB } from "../core/engine.js";
import { getLearningHealth, migrateRuleColumn } from "../core/qvalue.js";
import { llmKeyProblem } from "../core/llm.js";
import { TEMPLATE_PLACEHOLDER_LINKS } from "../core/promote.js";
import { slugify } from "../core/slug.js";

export type HealthResult = {
  success: boolean;
  data: Record<string, unknown>;
  warnings: string[];
};

export async function runHealth(
  startDir: string,
  linkGraph?: LinkGraph,
): Promise<HealthResult> {
  const vaultRoot = await findVaultRoot(startDir);
  const paths = getVaultPaths(vaultRoot);
  const config = await loadConfig(paths.config);

  const allNotes = await listNoteTitles(paths.notes);
  const graph = linkGraph ?? await buildGraph(paths.notes);
  const orphans = findOrphans(graph, allNotes);
  const dangling = findDanglingLinks(graph, allNotes);

  // #38: name the cause and the notes instead of leaving mystery dangling
  // links. Notes are the user's; health explains, it does not edit them.
  const placeholderSlugs = new Set(TEMPLATE_PLACEHOLDER_LINKS.map((t) => slugify(t)));
  const templateWarnings: string[] = [];
  for (const d of dangling) {
    if (!placeholderSlugs.has(slugify(d))) continue;
    const sources = [...(graph.incoming.get(d) ?? [])].sort();
    const shown = sources.slice(0, 10).join(", ");
    const more = sources.length > 10 ? ` and ${sources.length - 10} more` : "";
    templateWarnings.push(
      `[[${d}]] is likely a placeholder link from an older version of the note template, not a ` +
        `missing note. Found in ${sources.length} note(s): ${shown}${more}. If you did not write it, ` +
        "delete that link line (and from templates/note.md if it is still there). New notes no longer get it.",
    );
  }

  const schemaViolations: { note: string; errors: string[] }[] = [];
  const fading: { note: string; vitality: number }[] = [];

  for (const note of allNotes) {
    const filePath = path.join(paths.notes, `${note}.md`);
    const content = await fs.readFile(filePath, "utf8");
    const parsed = parseFrontmatter(content);
    const type =
      parsed.data && typeof parsed.data === "object"
        ? (parsed.data as Record<string, unknown>)["type"]
        : null;

    const templatePath = resolveTemplatePath(
      config,
      vaultRoot,
      typeof type === "string" ? type : null
    );
    const validation = await validateNoteAgainstSchema(filePath, templatePath);
    if (!validation.valid) {
      schemaViolations.push({ note, errors: validation.errors });
    }

    const dataObj =
      parsed.data && typeof parsed.data === "object"
        ? (parsed.data as Record<string, unknown>)
        : null;
    const lastAccessedRaw =
      typeof dataObj?.["last_accessed"] === "string"
        ? dataObj["last_accessed"]
        : typeof dataObj?.["created"] === "string"
          ? dataObj["created"]
          : null;
    if (typeof lastAccessedRaw === "string") {
      const last = new Date(lastAccessedRaw);
      if (!isNaN(last.getTime())) {
        const decayDays =
          typeof type === "string" && config.vitality.decay[type]
            ? config.vitality.decay[type]
            : 30;
        const vitality = computeVitality(
          { base: config.vitality.base, decayDays },
          last,
          new Date()
        );
        if (vitality < 0.2) {
          fading.push({ note, vitality });
        }
      }
    }
  }

  // Learning-signal health. Added 2026-08-28: a per-query reward proxy ran for
  // five months and inverted every Q-value, and nothing in `ori health` would
  // have shown it. These four numbers are the ones that would have.
  let learning: Record<string, unknown> | undefined;
  const learningWarnings: string[] = [];
  const dbPath = path.resolve(vaultRoot, config.engine.db_path);
  try {
    await fs.access(dbPath);
    const db = initDB(dbPath);
    try {
      // health can be the first command after upgrading; its queries need
      // note_q.rule, and a failure here was reported as "No index yet".
      migrateRuleColumn(db);
      const h = getLearningHealth(db);
      learning = { ...h };

      // Negative correlation between exposure and learned value is the
      // signature of the degenerate feedback loop. Measured at -0.537 during
      // the incident; anything below -0.1 warrants investigation.
      if (h.exposureQCorrelation < -0.1) {
        learningWarnings.push(
          `exposure/Q correlation is ${h.exposureQCorrelation.toFixed(3)} — ` +
            `frequently-used notes are being scored LOWER, which indicates a ` +
            `reward-signal defect, not a ranking preference.`,
        );
      }
      // Forward citation is the strongest signal in reward.ts. Zero of them
      // alongside real update traffic means key matching has broken again.
      if (h.totalUpdates > 200 && h.forwardCitations === 0) {
        learningWarnings.push(
          `0 forward citations across ${h.totalUpdates} Q-updates — the ` +
            `strongest reward signal is not firing; check note-key ` +
            `normalization in reward.ts buildOutcome().`,
        );
      }
      // >1 key shape means slug/title drift returned.
      if (h.distinctKeyShapes > 1) {
        learningWarnings.push(
          `note_q holds ${h.distinctKeyShapes} distinct key shapes — slug and ` +
            `raw-title ids have diverged, so Q-values are split per note.`,
        );
      }
      // Any source outside the sanctioned two means an unaudited writer.
      const unexpected = Object.keys(h.bySource).filter(
        (s) => s !== "session_batch" && s !== "explore_conclude",
      );
      if (unexpected.length > 0) {
        learningWarnings.push(
          `unexpected Q-update sources: ${unexpected.join(", ")}`,
        );
      }
      // The old "most notes never credited" check is gone (#37): CLI queries
      // record exposure without credit and neutral / dead-end outcomes write
      // nothing, so uncredited notes and sessions are normal. The outage it
      // guarded against (session-end flush never running) is reported per
      // session on stderr by serve.ts ("[ori] session learning: ...").
    } finally {
      db.close();
    }
  } catch {
    // No index yet — learning health is simply unavailable, not an error.
  }

  // Configured-but-unusable LLM (#42). Promote falls back to the deterministic
  // path, so the only downstream symptom is empty descriptions weeks later;
  // health is where people look when that happens.
  const configWarnings: string[] = [];
  const keyProblem = llmKeyProblem(config.llm);
  if (keyProblem) configWarnings.push(keyProblem);

  return {
    success: true,
    data: {
      noteCount: allNotes.length,
      orphanCount: orphans.length,
      danglingCount: dangling.length,
      orphans,
      dangling,
      schemaViolations,
      fading,
      ...(learning ? { learning } : {}),
    },
    warnings: [...templateWarnings, ...configWarnings, ...learningWarnings],
  };
}
