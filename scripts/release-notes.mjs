// Print the CHANGELOG section for a version (default: package.json's), without
// its "## [x.y.z]" heading. Used by .github/workflows/publish.yml for the
// GitHub release body. Exits 1 if the section is missing or empty.
import { readFileSync } from "node:fs";

const version =
  process.argv[2] ??
  JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const lines = readFileSync(new URL("../CHANGELOG.md", import.meta.url), "utf8").split(/\r?\n/);

const start = lines.findIndex((l) => l.startsWith(`## [${version}]`));
if (start === -1) {
  console.error(`CHANGELOG.md has no "## [${version}]" section.`);
  process.exit(1);
}
let end = lines.findIndex((l, i) => i > start && l.startsWith("## ["));
if (end === -1) end = lines.length;

const body = lines.slice(start + 1, end).join("\n").trim();
if (!body) {
  console.error(`CHANGELOG.md section for ${version} is empty.`);
  process.exit(1);
}
process.stdout.write(body + "\n");
