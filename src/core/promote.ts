import {
  classifyNoteType,
  detectProjects,
  type ClassificationResult,
  type ProjectKeywordConfig,
} from "./classify.js";
import {
  detectLinks,
  applyLinks,
  suggestLinks,
  isHub,
  type DetectedLink,
  type LinkSuggestion,
  type VaultIndex,
} from "./linkdetect.js";
import { slugify } from "./slug.js";

export type PromoteOverrides = {
  type?: string;
  description?: string;
  links?: string[];
  project?: string[];
};

export type PromoteInput = {
  inboxPath: string;
  frontmatter: Record<string, unknown>;
  body: string;
  existingTitles: string[];
  vaultIndex: VaultIndex;
  overrides: PromoteOverrides;
  projectConfig: ProjectKeywordConfig;
  mapRouting: Record<string, string>;
  defaultArea: string;
};

export type PromoteResult = {
  updatedFrontmatter: Record<string, unknown>;
  updatedBody: string;
  destinationFilename: string;
  classification: ClassificationResult;
  detectedLinks: DetectedLink[];
  suggestedLinks: LinkSuggestion[];
  suggestedAreas: string[];
  changes: string[];
  warnings: string[];
};

const AUTO_APPLY_THRESHOLD = 0.8;

const FOOTER_HEADINGS = ["Relevant Notes", "Areas"] as const;

/**
 * Placeholder links shipped by the note template before #38. Vaults created
 * earlier keep their own copy of the template; `ori add` strips these lines
 * from that template text before any user content goes in. Promote does NOT
 * filter them: at promote time a template line cannot be told from one the
 * user wrote, and user links are never removed.
 */
export const TEMPLATE_PLACEHOLDER_LINKS = ["related note", "relevant map"] as const;

/**
 * Remove the pre-#38 placeholder lines from TEMPLATE text (`- [[related note]]
 * -- ...`, `- [[relevant map]]`). Headings stay. Call this only on template
 * text, never on a body that contains user content.
 */
export function stripTemplatePlaceholderLines(templateText: string): string {
  return templateText
    .split("\n")
    .filter((line) => {
      const m = line.trim().match(/^-\s+\[\[([^\]|#]+)\]\]/);
      return !(m && (TEMPLATE_PLACEHOLDER_LINKS as readonly string[]).includes(m[1]!.trim().toLowerCase()));
    })
    .join("\n");
}

const TEMPLATE_PLACEHOLDER = /\{Content\s*[-—]/;

/**
 * Check if a note body still contains the unfilled template placeholder.
 * Used as a quality gate to prevent promoting empty stubs.
 */
export function isTemplatePlaceholder(body: string): boolean {
  return TEMPLATE_PLACEHOLDER.test(body);
}

/**
 * Parse an existing footer section from the body.
 * Looks for "## <heading>" or "<heading>:" followed by lines starting with "- ".
 */
type FooterItem = { title: string; line: string };

function parseFooter(body: string, heading: string): FooterItem[] {
  // Match both "## Areas" and "Areas:" formats, in EVERY section with that
  // heading: a note can carry its own footer plus an empty template one
  // (#46), and reading only the first would drop whichever came second.
  const patterns = [
    new RegExp(`^##\\s+${escapeRegex(heading)}\\s*$`, "gm"),
    new RegExp(`^${escapeRegex(heading)}:\\s*$`, "gm"),
  ];

  const items: FooterItem[] = [];
  for (const pattern of patterns) {
    for (const match of body.matchAll(pattern)) {
      const startIdx = (match.index ?? 0) + match[0].length;
      const remaining = body.slice(startIdx);
      const lines = remaining.split("\n");

      const knownHeadings = FOOTER_HEADINGS;
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("- ")) {
          // Title from "- [[title]]" or "- [[title]] -- reason". Keep the whole
          // line: the reason is the author's, and re-rendering from the title
          // alone deleted it on every promote (#46).
          const linkMatch = trimmed.match(/^-\s+\[\[([^\]]+)\]\]/);
          if (linkMatch) {
            items.push({ title: linkMatch[1], line: trimmed });
          }
        } else if (trimmed.length === 0) {
          continue;
        } else if (
          trimmed.startsWith("#") ||
          trimmed.startsWith("---") ||
          knownHeadings.some((h) => trimmed === `${h}:` || trimmed === `## ${h}`)
        ) {
          break;
        }
      }
    }
  }
  return items;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Strip existing footer sections (Areas and Relevant Notes) from body.
 * Line-based approach for reliability.
 */
function stripFooters(body: string): string {
  const headings = FOOTER_HEADINGS;
  const lines = body.split("\n");
  const kept: string[] = [];
  let inFooterSection = false;

  for (const line of lines) {
    const trimmed = line.trim();

    // Check if this line starts a footer section
    const isHeading = headings.some(
      (h) =>
        trimmed === `${h}:` ||
        trimmed === `## ${h}` ||
        trimmed.startsWith(`${h}:`)
    );

    if (isHeading) {
      inFooterSection = true;
      continue;
    }

    if (inFooterSection) {
      // Stay in footer section for list items and blank lines
      if (trimmed.startsWith("- ") || trimmed === "") {
        continue;
      }
      // Non-list, non-blank line exits the footer section
      inFooterSection = false;
    }

    kept.push(line);
  }

  return kept.join("\n").trimEnd();
}

/**
 * Format footer sections.
 */
function formatFooters(areas: string[], links: string[]): string {
  let footer = "";

  if (links.length > 0) {
    footer += "\n\nRelevant Notes:";
    for (const line of links) footer += `\n${line}`;
  }

  if (areas.length > 0) {
    footer += "\n\nAreas:";
    for (const line of areas) footer += `\n${line}`;
  }

  return footer + "\n";
}

/** Existing lines first (verbatim, first occurrence wins), then new titles. */
function mergeFooter(existing: FooterItem[], added: string[]): string[] {
  const seen = new Map<string, string>();
  for (const item of existing) if (!seen.has(item.title)) seen.set(item.title, item.line);
  for (const title of added) if (!seen.has(title)) seen.set(title, `- [[${title}]]`);
  return [...seen.values()];
}

/**
 * Inject footers with idempotency: parse existing, merge, dedupe, write once.
 */
export function injectFooters(
  body: string,
  areas: string[],
  links: string[]
): string {
  const existingAreas = parseFooter(body, "Areas");
  const existingLinks = parseFooter(body, "Relevant Notes");

  const mergedAreas = mergeFooter(existingAreas, areas);
  const mergedLinks = mergeFooter(existingLinks, links);

  const cleanBody = stripFooters(body);

  if (mergedAreas.length === 0 && mergedLinks.length === 0) {
    return cleanBody + "\n";
  }

  return cleanBody + formatFooters(mergedAreas, mergedLinks);
}

/**
 * Resolve areas for a note based on project tags and map routing config.
 * Fallback chain: config routing -> slug match on map titles -> defaultArea.
 *
 * Returns the areas plus whether the default had to be used, because the
 * default is not a map and a caller that cannot tell the difference will
 * report a note as filed when it is not.
 */
export function resolveAreas(
  projects: string[],
  mapRouting: Record<string, string>,
  existingTitles: string[],
  defaultArea: string
): { areas: string[]; usedDefault: boolean } {
  const areas: string[] = [];

  // Compare slugs, not raw strings. A project tag is hyphenated ("ai-agents")
  // and a map title is usually spaced ("ai agents map"), so the old
  // title.includes(project) test was false for the largest tag in a 1,548-note
  // vault -- 314 notes, every one of them silently routed to the default. The
  // map existed the whole time. Measured: 395 of 535 project-tagged slots
  // (73.8%) took the fallback, 314 of them from this one separator mismatch.
  const needles = projects.map((p) => ({ raw: p, slug: slugify(p) }));
  const haystacks = existingTitles
    .filter((t) => /(^|[-\s])map$/i.test(t.trim()))
    .map((t) => ({ title: t, slug: slugify(t) }));

  for (const { raw, slug } of needles) {
    if (mapRouting[raw]) {
      areas.push(mapRouting[raw]);
      continue;
    }
    const hit = haystacks.find((h) => h.slug.includes(slug));
    if (hit) areas.push(hit.title);
  }

  // Ensure at least one area -- zero orphans from promotion. This guard is
  // why the mismatch above stayed invisible: it turns "no map matched" into
  // an Areas footer that looks filled, so every downstream orphan check
  // passes while the note is filed under the hub rather than any map.
  // usedDefault is the signal that this happened.
  const usedDefault = areas.length === 0;
  if (usedDefault) areas.push(defaultArea);

  return { areas: [...new Set(areas)], usedDefault };
}

/**
 * Compute the promotion result without performing any I/O.
 * The caller (CLI or MCP) handles file move, validation, etc.
 */
export function computePromotion(input: PromoteInput): PromoteResult {
  const {
    inboxPath,
    frontmatter,
    body,
    existingTitles,
    vaultIndex,
    overrides,
    projectConfig,
    mapRouting,
    defaultArea,
  } = input;

  const changes: string[] = [];
  const warnings: string[] = [];

  // 1. Classify type
  const classification = classifyNoteType(
    titleFromPath(inboxPath),
    body,
    (overrides.type as string) ?? (frontmatter.type as string | undefined)
  );
  if (classification.confidence === "low" && !overrides.type) {
    warnings.push(
      `Low-confidence type classification: ${classification.type} (${classification.reason}). Use --type to override.`
    );
  }
  if (overrides.type && overrides.type !== frontmatter.type) {
    changes.push(`type: ${frontmatter.type ?? "unset"} → ${overrides.type}`);
  } else if (classification.type !== frontmatter.type) {
    changes.push(
      `type classified as ${classification.type} (${classification.confidence} confidence)`
    );
  }

  // 2. Detect projects
  let projects: string[];
  if (overrides.project && overrides.project.length > 0) {
    projects = overrides.project;
    changes.push(`project set to: ${projects.join(", ")}`);
  } else if (
    Array.isArray(frontmatter.project) &&
    frontmatter.project.length > 0
  ) {
    projects = frontmatter.project as string[];
  } else {
    projects = detectProjects(
      titleFromPath(inboxPath),
      body,
      projectConfig
    );
    if (projects.length > 0) {
      changes.push(`project detected: ${projects.join(", ")}`);
    } else {
      warnings.push("No project detected. Consider adding --project.");
    }
  }

  // 3. Detect wiki-links in body text. Not the default area or other hubs
  // (#44): "index" is a common word, and every note reaches it via Areas.
  const notLinkable = [defaultArea];
  const detectedLinks = detectLinks(
    body,
    existingTitles.filter(
      (t) => !notLinkable.includes(t) && !isHub(vaultIndex.graph, t)
    )
  );
  const unlinked = detectedLinks.filter((l) => !l.alreadyLinked);
  if (unlinked.length > 0) {
    changes.push(`auto-linked ${unlinked.length} mention(s) in body`);
  }

  // 4. Suggest structural links
  const allSuggested = suggestLinks(
    { ...frontmatter, project: projects },
    body,
    vaultIndex,
    { exclude: notLinkable }
  );
  // Auto-apply high-confidence suggestions
  const autoApplied = allSuggested.filter(
    (s) => s.confidence >= AUTO_APPLY_THRESHOLD
  );
  const manualSuggestions = allSuggested.filter(
    (s) => s.confidence < AUTO_APPLY_THRESHOLD
  );

  if (manualSuggestions.length > 0) {
    changes.push(
      `suggested ${manualSuggestions.length} connection(s): ${manualSuggestions.map((s) => s.title).join(", ")}`
    );
  }

  // 5. Apply links to body
  let updatedBody = applyLinks(body, detectedLinks);

  // Add override links if provided
  if (overrides.links && overrides.links.length > 0) {
    changes.push(
      `added ${overrides.links.length} explicit link(s): ${overrides.links.join(", ")}`
    );
  }

  // 6. Resolve areas
  const { areas: suggestedAreas, usedDefault: areaFellBack } = resolveAreas(
    projects,
    mapRouting,
    existingTitles,
    defaultArea
  );
  changes.push(`assigned to area(s): ${suggestedAreas.join(", ")}`);
  if (areaFellBack) {
    warnings.push(
      `No map matched ${projects.length > 0 ? `project(s) ${projects.join(", ")}` : "this note"}; ` +
        `filed under "${defaultArea}", which is a hub and not a map. Add a ` +
        `promote.project_map_routing entry or create the map. The note will ` +
        `pass orphan checks without belonging to any map.`
    );
  }

  // 7. Inject footers (idempotent)
  const allLinks = [
    ...autoApplied.map((s) => s.title),
    ...(overrides.links ?? []),
  ];
  updatedBody = injectFooters(updatedBody, suggestedAreas, allLinks);

  // 8. Update frontmatter
  const updatedFrontmatter: Record<string, unknown> = {
    ...frontmatter,
    status: "active",
    type: classification.type,
    project: projects.length > 0 ? projects : frontmatter.project,
    last_accessed: new Date().toISOString().split("T")[0],
    access_count:
      (typeof frontmatter.access_count === "number"
        ? frontmatter.access_count
        : 0) + 1,
  };

  if (overrides.description) {
    updatedFrontmatter.description = overrides.description;
    changes.push("description updated via override");
  } else if (
    !frontmatter.description ||
    (typeof frontmatter.description === "string" &&
      frontmatter.description.trim().length === 0)
  ) {
    warnings.push(
      'No description found. Add with --description "..." or configure LLM.'
    );
  }

  changes.push("status: inbox → active");

  // 9. Derive destination filename
  const filename = inboxPath.split(/[/\\]/).pop() ?? "note.md";
  const destinationFilename = filename.endsWith(".md")
    ? filename
    : `${filename}.md`;

  return {
    updatedFrontmatter,
    updatedBody,
    destinationFilename,
    classification,
    detectedLinks,
    suggestedLinks: allSuggested,
    suggestedAreas,
    changes,
    warnings,
  };
}

function titleFromPath(filePath: string): string {
  const filename = filePath.split(/[/\\]/).pop() ?? "";
  return filename.replace(/\.md$/, "").replace(/-/g, " ");
}
