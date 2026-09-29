/**
 * #38 end to end: a vault created before the fix keeps the old template (with
 * live placeholder links). New notes must still come out with no dangling
 * links, inbox notes included, a real note named like a placeholder must keep
 * its links, and health must name the cause of any placeholders already there.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import { runInit } from "../../src/cli/init.js";
import { runAdd } from "../../src/cli/add.js";
import { runHealth } from "../../src/cli/health.js";

const OLD_FOOTER =
  "Relevant Notes:\n- [[related note]] -- why this connection matters\n\nAreas:\n- [[relevant map]]\n";

let vault: string;
beforeEach(async () => {
  vault = await fs.mkdtemp(path.join(os.tmpdir(), "ori-38-"));
  await runInit({ targetDir: vault });
  const tpl = path.join(vault, "templates", "note.md");
  const cur = await fs.readFile(tpl, "utf8");
  // Recreate the pre-#38 template footer.
  await fs.writeFile(tpl, cur.replace(/Relevant Notes:[\s\S]*$/, OLD_FOOTER), "utf8");
});
afterEach(async () => {
  await fs.rm(vault, { recursive: true, force: true });
});

const read = (p: string) => fs.readFile(p, "utf8");

describe("#38 with a pre-fix template", () => {
  it("a promoted note has no placeholder links and health reports none dangling", async () => {
    const r = await runAdd({ startDir: vault, title: "citing is the strongest use signal", content: "Real content." });
    expect(String(r.data.path)).toContain(`${path.sep}notes${path.sep}`); // really promoted
    const body = await read(r.data.path as string);
    expect(body).not.toMatch(/\[\[(related note|relevant map)\]\]/i);
    const h = await runHealth(vault);
    expect(h.data.dangling).toEqual([]);
  });

  it("an inbox note that is never promoted has no placeholder links either", async () => {
    const r = await runAdd({ startDir: vault, title: "a note with no content yet" });
    expect(String(r.data.path)).toContain(`${path.sep}inbox${path.sep}`);
    expect(await read(r.data.path as string)).not.toMatch(/\[\[(related note|relevant map)\]\]/i);
  });

  it("does not link new notes to a real note that happens to share a placeholder's name", async () => {
    await runAdd({ startDir: vault, title: "relevant map", content: "A real map note." });
    const r = await runAdd({ startDir: vault, title: "a note under some other map", content: "Content." });
    // The line came from the template, not the user: stripped regardless.
    expect(await read(r.data.path as string)).not.toContain("[[relevant map]]");
  });

  it("a footer link the user wrote survives promotion, even if it is placeholder-named", async () => {
    const r = await runAdd({
      startDir: vault,
      title: "user plans a map",
      content: "Plan.\n\nAreas:\n- [[relevant map]]\n\nRelevant Notes:\n- [[related note]] -- write this next",
    });
    expect(String(r.data.path)).toContain(`${path.sep}notes${path.sep}`);
    const body = await read(r.data.path as string);
    expect(body).toContain("[[relevant map]]");
    expect(body).toContain("[[related note]]");
  });

  it("health names the old template as the cause of existing placeholder links", async () => {
    await fs.writeFile(
      path.join(vault, "notes", "legacy.md"),
      "---\ndescription: x\ntype: insight\nproject: []\nstatus: active\ncreated: 2026-01-01\n---\n# legacy\n\n" + OLD_FOOTER,
      "utf8",
    );
    const h = await runHealth(vault);
    expect(h.data.dangling).toEqual(expect.arrayContaining(["related-note", "relevant-map"]));
    const w = h.warnings.find((x) => x.includes("older version of the note template"));
    expect(w).toBeDefined();
    expect(w).toContain("[[related-note]]");
    expect(w).toContain("legacy"); // names the note that carries it
  });

  it("never touches a placeholder-looking line the user wrote in their own content", async () => {
    const r = await runAdd({
      startDir: vault,
      title: "a plan with a map to create later",
      content: "Plan.\n- [[relevant map]] -- to create",
    });
    expect(await read(r.data.path as string)).toContain("- [[relevant map]] -- to create");
  });

  it("health does not call a link a placeholder when a real note has that name", async () => {
    await runAdd({ startDir: vault, title: "relevant map", content: "A real map note." });
    await fs.writeFile(
      path.join(vault, "notes", "legacy.md"),
      "---\ndescription: x\ntype: insight\nproject: []\nstatus: active\ncreated: 2026-01-01\n---\n# legacy\n\n" + OLD_FOOTER,
      "utf8",
    );
    const h = await runHealth(vault);
    expect(h.warnings.some((x) => x.includes("[[relevant-map]]"))).toBe(false);
  });
});
