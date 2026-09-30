/**
 * "What's new" after an update.
 *
 * The update notice (update-check.ts) covers the moment BEFORE an update. This
 * covers the moment after: the first session on a newer version shows the
 * agent the CHANGELOG sections since the version this machine last ran, once.
 * Keyed on the last version seen, not on the publish date, so it fires whether
 * the user updates a day or two months after release, and a skipped version's
 * notes are included.
 *
 * CHANGELOG.md ships in the npm package, so this works offline.
 */
import { promises as fs, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { VERSION } from "./version.js";
import { compareVersions, getUpdateCachePath } from "./update-check.js";

/** Longest notice attached to a tool response; the rest is one tool call away. */
const MAX_NOTICE_CHARS = 4000;

export interface ChangelogSection {
  version: string;
  body: string;
}

/** CHANGELOG.md next to package.json: dist/core -> ../../, src/core -> ../../. */
export function readChangelog(): string | null {
  const here = path.dirname(fileURLToPath(import.meta.url));
  try {
    return readFileSync(path.join(here, "..", "..", "CHANGELOG.md"), "utf8");
  } catch {
    return null;
  }
}

/** Split on `## [x.y.z]` headings. Sections come back in file order (newest first). */
export function parseChangelog(text: string): ChangelogSection[] {
  const sections: ChangelogSection[] = [];
  const heading = /^## \[(\d+\.\d+\.\d+)\][^\n]*$/gm;
  const marks = [...text.matchAll(heading)];
  marks.forEach((m, i) => {
    const end = i + 1 < marks.length ? marks[i + 1]!.index : text.length;
    sections.push({ version: m[1]!, body: text.slice(m.index, end).trim() });
  });
  return sections;
}

/** Sections newer than `since` and no newer than `upTo`, newest first. */
export function sectionsBetween(
  sections: ChangelogSection[],
  since: string,
  upTo: string,
): ChangelogSection[] {
  return sections.filter(
    (s) => compareVersions(since, s.version) && !compareVersions(upTo, s.version),
  );
}

function lastSeenPath(): string {
  return path.join(path.dirname(getUpdateCachePath()), "last-seen-version.json");
}

export async function readLastSeen(): Promise<string | null> {
  try {
    const parsed = JSON.parse(await fs.readFile(lastSeenPath(), "utf8")) as { version?: string };
    return typeof parsed.version === "string" ? parsed.version : null;
  } catch {
    return null;
  }
}

export async function writeLastSeen(version: string): Promise<void> {
  const file = lastSeenPath();
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, JSON.stringify({ version, seenAt: new Date().toISOString() }), "utf8");
}

/** True if an earlier Ori ran on this machine (it leaves an update-check cache). */
async function hadPriorInstall(): Promise<boolean> {
  try {
    await fs.access(getUpdateCachePath());
    return true;
  } catch {
    return false;
  }
}

/**
 * The notice for this session, or null. Marks the current version as seen when
 * it returns one, so the notice is shown once per machine per update.
 *
 * No last-seen record: a brand-new install gets nothing (there is nothing
 * "new" to a first-time user) and is recorded; an install upgraded from a
 * version that predates this feature gets the current version's notes.
 */
export async function takeWhatsNewNotice(
  current: string = VERSION,
  changelog: string | null = readChangelog(),
): Promise<string | null> {
  const lastSeen = await readLastSeen();
  if (lastSeen && !compareVersions(lastSeen, current)) return null;

  if (!lastSeen && !(await hadPriorInstall())) {
    await writeLastSeen(current);
    return null;
  }

  const sections = changelog ? parseChangelog(changelog) : [];
  const shown = lastSeen
    ? sectionsBetween(sections, lastSeen, current)
    : sections.filter((s) => s.version === current);
  await writeLastSeen(current);
  if (shown.length === 0) return null;

  let notes = shown.map((s) => s.body).join("\n\n");
  if (notes.length > MAX_NOTICE_CHARS) {
    notes = `${notes.slice(0, MAX_NOTICE_CHARS)}\n\n[truncated; call ori_whats_new for the rest]`;
  }
  return (
    `Ori was updated${lastSeen ? ` from v${lastSeen}` : ""} to v${current}. ` +
    `Tell the user briefly, in your own words, what changed for them. Release notes:\n\n${notes}`
  );
}

/** Notes for one version (default: the installed one), for ori_whats_new / `ori whats-new`. */
export function whatsNew(version: string = VERSION, changelog: string | null = readChangelog()): string {
  const section = changelog ? parseChangelog(changelog).find((s) => s.version === version) : undefined;
  return section?.body ?? `No release notes found for v${version}.`;
}
