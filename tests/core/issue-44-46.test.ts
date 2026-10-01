/**
 * #44 auto-linking inside code and on hub titles.
 * #45 suggested connections collapse to the alphabetical head through a hub.
 * #46 complete-note content: duplicate H1, template trailer, dropped footer reasons.
 */
import { describe, it, expect } from "vitest";
import { maskCode } from "../../src/core/graph.js";
import { detectLinks, suggestLinks, HUB_DEGREE, type VaultIndex } from "../../src/core/linkdetect.js";
import { computePromotion, injectFooters, type PromoteInput } from "../../src/core/promote.js";
import { composeBody } from "../../src/cli/add.js";

function index(titles: string[], incoming: Record<string, string[]> = {}, outgoing: Record<string, string[]> = {}): VaultIndex {
  const toMap = (o: Record<string, string[]>) => new Map(Object.entries(o).map(([k, v]) => [k, new Set(v)]));
  return { titles, frontmatter: new Map(), graph: { incoming: toMap(incoming), outgoing: toMap(outgoing) } };
}

function promote(body: string, vaultIndex: VaultIndex): ReturnType<typeof computePromotion> {
  const input: PromoteInput = {
    inboxPath: "inbox/my-note.md",
    frontmatter: { description: "d", type: "insight", project: [], status: "inbox", created: "2026-10-01" },
    body,
    existingTitles: vaultIndex.titles,
    vaultIndex,
    overrides: {},
    projectConfig: { known_projects: [], keywords: {} },
    mapRouting: {},
    defaultArea: "index",
  };
  return computePromotion(input);
}

const hubLinkers = Array.from({ length: HUB_DEGREE + 1 }, (_, i) => `note-${String(i).padStart(3, "0")}`);

describe("#44 maskCode", () => {
  it("keeps length and newlines, blanks fences and inline spans", () => {
    const src = "a `x y` b\n```\ncode here\n```\nc";
    const m = maskCode(src);
    expect(m.length).toBe(src.length);
    expect(m).toBe("a       b\n   \n         \n   \nc");
  });
});

describe("#44 auto-linking", () => {
  it("does not link a title inside fenced or inline code", () => {
    const body = "```\n.ori/caching strategy.db\n```\nSee `caching strategy` and the caching strategy itself.";
    const links = detectLinks(body, ["caching strategy"]);
    expect(links).toHaveLength(1);
    expect(body.slice(links[0].offset, links[0].offset + links[0].length)).toBe("caching strategy");
    expect(links[0].offset).toBe(body.lastIndexOf("caching strategy"));
  });

  it("promote never links the default area from prose or code", () => {
    const vi = index(["index", "caching strategy"]);
    const r = promote("each vault index is small\n```\n.ori/index.db\n```\nsee caching strategy", vi);
    const prose = r.updatedBody.split("Relevant Notes:")[0]!.split("Areas:")[0]!;
    expect(prose).not.toContain("[[index]]");
    expect(prose).toContain(".ori/index.db");
    expect(prose).toContain("[[caching strategy]]");
  });

  it("promote does not auto-link a hub title", () => {
    const vi = index(["big hub", ...hubLinkers], { "big hub": hubLinkers });
    const r = promote("this mentions the big hub once", vi);
    expect(r.updatedBody).not.toContain("[[big hub]] once");
  });
});

describe("#45 suggested connections", () => {
  it("a hub is not a triangle pivot, so its linkers are not suggested", () => {
    const vi = index(["big hub", ...hubLinkers], { "big hub": hubLinkers });
    const s = suggestLinks({}, "we discuss big hub here", vi);
    expect(s.filter((x) => x.reason === "shared-neighborhood")).toEqual([]);
  });

  it("ranks candidates by how many of the note's links they share, then by title", () => {
    const vi = index(
      ["alpha topic", "beta topic", "zeta both", "aaa only alpha"],
      { "alpha topic": ["zeta both", "aaa only alpha"], "beta topic": ["zeta both"] },
    );
    const s = suggestLinks({}, "on alpha topic and beta topic", vi).filter((x) => x.reason === "shared-neighborhood");
    expect(s.map((x) => x.title)).toEqual(["zeta both", "aaa only alpha"]);
  });

  it("honours caller exclusions", () => {
    const vi = index(["index", "alpha topic", "x"], { "alpha topic": ["x"] });
    const s = suggestLinks({}, "index and alpha topic", vi, { exclude: ["index"] });
    expect(s.map((x) => x.title)).not.toContain("index");
  });
});

describe("#46 footers keep the author's reasons", () => {
  it("re-rendering keeps '-- reason' text and merges every footer section", () => {
    const body = [
      "# T", "", "text", "",
      "Relevant Notes:", "- [[a]] -- why a matters", "",
      "---", "", "Relevant Notes:", "", "Areas:", "- [[some map]]", "",
    ].join("\n");
    const out = injectFooters(body, ["other map"], ["b"]);
    expect(out).toContain("- [[a]] -- why a matters");
    expect(out).toContain("- [[b]]");
    expect(out.match(/Relevant Notes:/g)).toHaveLength(1);
    expect(out).toContain("- [[some map]]");
    expect(out).toContain("- [[other map]]");
  });
});

describe("#46 composeBody", () => {
  const template = "# {prose-as-title: x}\n\n{Content — y}\n\n---\n\n<!-- fill -->\n\nRelevant Notes:\n\nAreas:\n";

  it("plain content goes into the template under the given title", () => {
    expect(composeBody(template, "# My Title", "hello")).toBe(
      "# My Title\n\nhello\n\n---\n\n<!-- fill -->\n\nRelevant Notes:\n\nAreas:\n",
    );
  });

  it("content with its own H1 replaces the template title instead of duplicating it", () => {
    const out = composeBody(template, "# My Title", "# Their Title\n\nbody");
    expect(out.match(/^# /gm)).toHaveLength(1);
    expect(out.startsWith("# Their Title\n\nbody")).toBe(true);
  });

  it("content with its own footer is the whole body, no template trailer", () => {
    const content = "# Their Title\n\nbody\n\nRelevant Notes:\n- [[a]] -- why\n\nAreas:\n- [[m]]";
    const out = composeBody(template, "# My Title", content);
    expect(out).toBe(content + "\n");
    expect(out).not.toContain("<!-- fill -->");
  });

  it("footer without H1 gets the given title", () => {
    expect(composeBody(template, "# My Title", "body\n\nAreas:\n- [[m]]")).toBe("# My Title\n\nbody\n\nAreas:\n- [[m]]\n");
  });

  it("dollar patterns in content are literal", () => {
    expect(composeBody(template, "# T", "costs $& and $1")).toContain("costs $& and $1");
  });
});
