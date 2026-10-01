/**
 * #40: inline code spans must not produce wikilinks.
 * #41: note_q must not hold rows for things that are not notes.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import Database from "better-sqlite3";
import { stripCodeFences, stripInlineCode } from "../../src/core/graph.js";
import { initIndexStore, syncIndex } from "../../src/core/indexstore.js";
import {
  initQValueTables, incrementExposure, updateQ, getExposureCount, isKnownNote,
} from "../../src/core/qvalue.js";

const links = (s: string): string[] =>
  [...stripCodeFences(s).matchAll(/\[\[([^\]]+)\]\]/g)].map((m) => m[1]!);

describe("#40 inline code spans", () => {
  it("ignores bash test syntax in single-backtick spans", () => {
    expect(links("Use `[[ -t 0 && -t 1 ]]` to check, see [[real note]].")).toEqual(["real note"]);
  });

  it("handles double-backtick spans containing a backtick", () => {
    expect(links("``a ` [[ -w /dev/tty ]]`` then [[x]]")).toEqual(["x"]);
  });

  it("a run only closes on the same length", () => {
    expect(links("``[[a]]` still code [[b]]`` [[c]]")).toEqual(["c"]);
  });

  it("an unclosed backtick is literal and does not eat links", () => {
    expect(links("it`s [[kept]]")).toEqual(["kept"]);
  });

  it("a span does not cross a blank line", () => {
    expect(links("stray ` here\n\n[[kept]] and `x`")).toEqual(["kept"]);
  });

  it("spans may wrap within a paragraph, newlines preserved", () => {
    const out = stripInlineCode("a `[[x\ny]]` b");
    expect(out).toBe("a  \n b");
  });

  it("fences are still stripped first", () => {
    expect(links("```\n`[[in-fence]]`\n```\n[[after]]")).toEqual(["after"]);
  });
});

describe("#41 note_q only for real notes", () => {
  let root: string;
  let notesDir: string;
  let db: InstanceType<typeof Database>;

  beforeEach(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ori-41-"));
    notesDir = path.join(root, "notes");
    await fs.mkdir(notesDir);
    db = new Database(":memory:");
    initIndexStore(db);
    initQValueTables(db);
  });

  afterEach(async () => {
    db.close();
    await fs.rm(root, { recursive: true, force: true });
  });

  it("allows everything when the index is empty (degraded mode)", () => {
    expect(isKnownNote(db, "anything")).toBe(true);
    incrementExposure(db, "anything");
    expect(getExposureCount(db, "anything")).toBe(1);
  });

  it("refuses exposure and credit for non-notes once indexed", async () => {
    await fs.writeFile(path.join(notesDir, "Real Note.md"), "---\n---\n\nsee [[a-note-nobody-wrote]]\n");
    await syncIndex(db, notesDir);

    incrementExposure(db, "Real Note");
    incrementExposure(db, "a-note-nobody-wrote");
    updateQ(db, "a-note-nobody-wrote", 1, "s1");

    const ids = (db.prepare("SELECT note_id FROM note_q").all() as { note_id: string }[]).map((r) => r.note_id);
    expect(ids).toEqual(["real-note"]);
  });

  it("sync prunes never-credited phantom rows and keeps credited ones", async () => {
    db.prepare("INSERT INTO note_q (note_id, exposure_count) VALUES ('phantom', 3)").run();
    db.prepare("INSERT INTO note_q (note_id, update_count, q_value) VALUES ('earned', 2, 0.5)").run();
    await fs.writeFile(path.join(notesDir, "Real Note.md"), "---\n---\n\nbody\n");
    await syncIndex(db, notesDir);

    const ids = (db.prepare("SELECT note_id FROM note_q ORDER BY note_id").all() as { note_id: string }[]).map((r) => r.note_id);
    expect(ids).toEqual(["earned"]);
  });
});
