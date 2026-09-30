import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import {
  parseChangelog,
  sectionsBetween,
  takeWhatsNewNotice,
  whatsNew,
  readLastSeen,
  writeLastSeen,
  readChangelog,
} from "../../src/core/whats-new.js";
import { getUpdateCachePath } from "../../src/core/update-check.js";

const LOG = `# Changelog

## [0.9.0] - 2026-11-01

Nine.

## [0.8.0] - 2026-10-15

Eight.

## [0.7.1] - 2026-10-01

Seven-one.
`;

let dir: string;
beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "ori-whatsnew-"));
  process.env.ORI_UPDATE_CACHE_DIR = dir;
});
afterEach(async () => {
  delete process.env.ORI_UPDATE_CACHE_DIR;
  await fs.rm(dir, { recursive: true, force: true });
});

/** What any earlier Ori leaves behind: the update-check cache. */
async function priorInstall(): Promise<void> {
  const cache = getUpdateCachePath();
  await fs.mkdir(path.dirname(cache), { recursive: true });
  await fs.writeFile(cache, JSON.stringify({ latest: "0.7.0", checkedAt: 0 }));
}

describe("changelog parsing", () => {
  it("splits on version headings, newest first", () => {
    expect(parseChangelog(LOG).map((s) => s.version)).toEqual(["0.9.0", "0.8.0", "0.7.1"]);
  });

  it("selects the versions after `since` up to `upTo`", () => {
    const got = sectionsBetween(parseChangelog(LOG), "0.7.1", "0.9.0").map((s) => s.version);
    expect(got).toEqual(["0.9.0", "0.8.0"]);
  });

  it("the shipped CHANGELOG parses", () => {
    const text = readChangelog();
    expect(text).not.toBeNull();
    expect(parseChangelog(text!).length).toBeGreaterThan(0);
  });
});

describe("what's new notice", () => {
  it("a brand-new install gets no notice, and is recorded", async () => {
    expect(await takeWhatsNewNotice("0.7.1", LOG)).toBeNull();
    expect(await readLastSeen()).toBe("0.7.1");
  });

  it("an upgrade from a version before this feature shows the current notes", async () => {
    await priorInstall();
    const n = await takeWhatsNewNotice("0.7.1", LOG);
    expect(n).toContain("updated to v0.7.1");
    expect(n).toContain("Seven-one.");
  });

  it("shows every skipped version, however late the update", async () => {
    await writeLastSeen("0.7.1");
    const n = await takeWhatsNewNotice("0.9.0", LOG)!;
    expect(n).toContain("from v0.7.1 to v0.9.0");
    expect(n).toContain("Nine.");
    expect(n).toContain("Eight.");
    expect(n).not.toContain("Seven-one.");
  });

  it("is shown once", async () => {
    await writeLastSeen("0.8.0");
    expect(await takeWhatsNewNotice("0.9.0", LOG)).not.toBeNull();
    expect(await takeWhatsNewNotice("0.9.0", LOG)).toBeNull();
  });

  it("says nothing on the same or an older version", async () => {
    await writeLastSeen("0.9.0");
    expect(await takeWhatsNewNotice("0.9.0", LOG)).toBeNull();
    expect(await takeWhatsNewNotice("0.8.0", LOG)).toBeNull();
  });

  it("caps long notes and points at the tool", async () => {
    await writeLastSeen("0.7.1");
    const long = `## [0.8.0] - x\n\n${"word ".repeat(2000)}`;
    const n = (await takeWhatsNewNotice("0.8.0", long))!;
    expect(n.length).toBeLessThan(4400);
    expect(n).toContain("call ori_whats_new");
  });
});

describe("whatsNew", () => {
  it("returns one version's notes, or says there are none", () => {
    expect(whatsNew("0.8.0", LOG)).toContain("Eight.");
    expect(whatsNew("0.1.0", LOG)).toBe("No release notes found for v0.1.0.");
  });
});
