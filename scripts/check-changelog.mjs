// Refuse to publish a version the CHANGELOG does not describe: the installed
// package shows these notes to users after they update (src/core/whats-new.ts),
// and a published package cannot be amended.
import { readFileSync } from "node:fs";
const { version } = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const log = readFileSync(new URL("../CHANGELOG.md", import.meta.url), "utf8");
// Escape with "\\" - in a template literal "\[" and "\." are just "[" and ".",
// which made this a character class that could never match a real heading.
const escaped = version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
if (!new RegExp(`^## \\[${escaped}\\]`, "m").test(log)) {
  console.error(`CHANGELOG.md has no "## [${version}]" section. Users see these notes after updating; add it before publishing.`);
  process.exit(1);
}
console.log(`CHANGELOG.md describes ${version}.`);
