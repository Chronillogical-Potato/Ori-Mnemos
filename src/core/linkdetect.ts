import { maskCode, type LinkGraph } from "./graph.js";

/**
 * A note with more incoming links than this is a hub (an index, a top-level
 * map). Hubs are never auto-linked from prose and never used as a
 * triangle-closing pivot (#44, #45): every note links them, so they carry no
 * signal about what THIS note relates to. 128 matches the bootstrap poster
 * cap for the same reason.
 */
export const HUB_DEGREE = 128;

export function isHub(graph: LinkGraph, title: string, cap = HUB_DEGREE): boolean {
  return (graph.incoming.get(title)?.size ?? 0) > cap;
}

export type DetectedLink = {
  title: string;
  offset: number;
  length: number;
  alreadyLinked: boolean;
};

export type LinkSuggestion = {
  title: string;
  reason:
    | "title-match"
    | "tag-overlap"
    | "project-overlap"
    | "shared-neighborhood"
    | "semantic-similarity";
  confidence: number;
};

export type VaultIndex = {
  titles: string[];
  frontmatter: Map<string, Record<string, unknown>>;
  graph: LinkGraph;
};

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Build a regex that matches a note title in body text.
 * Slug-aware: dashes in titles also match spaces, and vice versa.
 */
function titleToPattern(title: string): RegExp {
  // Replace dashes with a pattern that matches dash or space
  const flexible = escapeRegex(title).replace(/-/g, "[-\\s]");
  return new RegExp(`\\b${flexible}\\b`, "gi");
}

/**
 * Check if the match at `offset` is already inside [[...]]
 */
function isInsideWikiLink(body: string, offset: number): boolean {
  // Walk backwards from offset looking for [[ without ]]
  let i = offset - 1;
  while (i >= 1) {
    if (body[i] === "[" && body[i - 1] === "[") {
      // Found [[ before this position — check no ]] between [[ and offset
      const between = body.slice(i + 1, offset);
      if (!between.includes("]]")) {
        return true;
      }
    }
    if (body[i] === "]" && i > 0 && body[i - 1] === "]") {
      // Found ]] before reaching [[, so we're not inside a link
      break;
    }
    i--;
  }
  return false;
}

/**
 * Scan body text for mentions of existing note titles.
 * Returns detected mentions sorted by offset.
 * Skips mentions already wrapped in [[]], and anything inside fenced or
 * inline code (#44): a title inside a code sample is not a reference, and
 * applyLinks would rewrite the code.
 */
export function detectLinks(
  body: string,
  existingTitles: string[]
): DetectedLink[] {
  // Offsets into the mask are offsets into body; code is blank in the mask.
  const text = maskCode(body);
  // Sort longest first to avoid partial matches
  const sorted = [...existingTitles].sort((a, b) => b.length - a.length);
  const results: DetectedLink[] = [];
  const covered = new Set<number>(); // track covered character positions

  for (const title of sorted) {
    if (title.length === 0) continue;
    const pattern = titleToPattern(title);
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(text)) !== null) {
      const offset = match.index;
      const length = match[0].length;

      // Skip if any position in this range is already covered
      let overlaps = false;
      for (let p = offset; p < offset + length; p++) {
        if (covered.has(p)) {
          overlaps = true;
          break;
        }
      }
      if (overlaps) continue;

      const alreadyLinked = isInsideWikiLink(body, offset);
      results.push({ title, offset, length, alreadyLinked });

      // Mark positions as covered
      for (let p = offset; p < offset + length; p++) {
        covered.add(p);
      }
    }
  }

  return results.sort((a, b) => a.offset - b.offset);
}

/**
 * Apply detected links to body text, wrapping unlinked mentions in [[]].
 * Processes from end to start to preserve offsets.
 */
export function applyLinks(body: string, links: DetectedLink[]): string {
  // Filter to only unlinked, sort by offset descending
  const toApply = links
    .filter((l) => !l.alreadyLinked)
    .sort((a, b) => b.offset - a.offset);

  let result = body;
  for (const link of toApply) {
    const before = result.slice(0, link.offset);
    const after = result.slice(link.offset + link.length);
    result = `${before}[[${link.title}]]${after}`;
  }
  return result;
}

/**
 * Suggest structural connections via graph heuristics.
 * No LLM — pure computation over vault metadata and link graph.
 */
export function suggestLinks(
  frontmatter: Record<string, unknown>,
  body: string,
  vaultIndex: VaultIndex,
  opts: { exclude?: Iterable<string> } = {}
): LinkSuggestion[] {
  const excluded = new Set(opts.exclude ?? []);
  const linkable = linkableTitles(vaultIndex, excluded);
  const suggestions = new Map<string, LinkSuggestion>();
  const noteProject = Array.isArray(frontmatter.project)
    ? (frontmatter.project as string[])
    : [];
  const noteTags = Array.isArray(frontmatter.tags)
    ? (frontmatter.tags as string[])
    : [];

  // Title match suggestions (from detectLinks)
  const detected = detectLinks(body, linkable);
  for (const link of detected) {
    if (!link.alreadyLinked) {
      suggestions.set(link.title, {
        title: link.title,
        reason: "title-match",
        confidence: 0.9,
      });
    }
  }

  // Project overlap (skip when a project has too many notes to be a useful signal)
  if (noteProject.length > 0) {
    const projectSizes = new Map<string, number>();
    for (const [, fm] of vaultIndex.frontmatter) {
      const projects = Array.isArray(fm.project) ? (fm.project as string[]) : [];
      for (const p of projects)
        projectSizes.set(p, (projectSizes.get(p) ?? 0) + 1);
    }

    for (const [title, fm] of vaultIndex.frontmatter) {
      if (suggestions.has(title)) continue;
      const otherProject = Array.isArray(fm.project)
        ? (fm.project as string[])
        : [];
      const overlap = noteProject.filter(
        (p) => otherProject.includes(p) && (projectSizes.get(p) ?? 0) <= 10,
      );
      if (overlap.length > 0) {
        suggestions.set(title, {
          title,
          reason: "project-overlap",
          confidence: 0.6 + overlap.length * 0.1,
        });
      }
    }
  }

  // Tag overlap
  if (noteTags.length > 0) {
    for (const [title, fm] of vaultIndex.frontmatter) {
      if (suggestions.has(title)) continue;
      const otherTags = Array.isArray(fm.tags) ? (fm.tags as string[]) : [];
      const overlap = noteTags.filter((t) => otherTags.includes(t));
      if (overlap.length > 0) {
        suggestions.set(title, {
          title,
          reason: "tag-overlap",
          confidence: 0.5 + overlap.length * 0.1,
        });
      }
    }
  }

  // Shared neighborhood (triangle closing)
  // If the new note links to X, and Y also links to X (or X links to Y),
  // suggest Y as a connection.
  //
  // Hubs are skipped as pivots (#45): a hub's ~every-note neighborhood gave
  // hundreds of candidates at one confidence, and the tie kept insertion
  // order, so the suggestion was the alphabetical head of the vault. Among
  // the rest, a candidate reached through more of this note's links ranks
  // higher (`shared`), and title breaks any remaining tie deterministically.
  const myLinks = new Set(
    detected.filter((d) => !d.alreadyLinked).map((d) => d.title)
  );
  const shared = new Map<string, number>();
  const consider = (title: string, confidence: number): void => {
    if (myLinks.has(title) || excluded.has(title)) return;
    const prev = suggestions.get(title);
    if (prev && prev.reason !== "shared-neighborhood") return;
    shared.set(title, (shared.get(title) ?? 0) + 1);
    if (!prev || prev.confidence < confidence) {
      suggestions.set(title, { title, reason: "shared-neighborhood", confidence });
    }
  };
  for (const linkedTitle of myLinks) {
    if (isHub(vaultIndex.graph, linkedTitle)) continue;
    // Other notes that also link to linkedTitle
    for (const coLinker of vaultIndex.graph.incoming.get(linkedTitle) ?? []) {
      consider(coLinker, 0.5);
    }
    // Notes that linkedTitle links to
    for (const target of vaultIndex.graph.outgoing.get(linkedTitle) ?? []) {
      consider(target, 0.45);
    }
  }

  return Array.from(suggestions.values())
    .sort(
      (a, b) =>
        b.confidence - a.confidence ||
        (shared.get(b.title) ?? 0) - (shared.get(a.title) ?? 0) ||
        a.title.localeCompare(b.title)
    )
    .slice(0, 5);
}

/**
 * Titles that may be auto-linked from prose: not excluded by the caller and
 * not a hub (#44). The default area (usually `index`) is passed in as an
 * exclusion by promote; its name is a common word and every note already
 * reaches it through the Areas footer.
 */
export function linkableTitles(
  vaultIndex: VaultIndex,
  exclude: Iterable<string> = []
): string[] {
  const excluded = new Set(exclude);
  return vaultIndex.titles.filter(
    (t) => !excluded.has(t) && !isHub(vaultIndex.graph, t)
  );
}

