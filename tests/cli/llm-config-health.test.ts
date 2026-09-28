/**
 * #42: a configured LLM whose key does not resolve must be reported, not
 * silently degraded to NullProvider. Before the fix the only symptom was
 * promote's "configure LLM" hint - which blames the user for not doing
 * something they did - while notes accumulated empty descriptions.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import YAML from "yaml";
import { runHealth } from "../../src/cli/health.js";
import { runInit } from "../../src/cli/init.js";
import { createProvider, llmKeyProblem, NullProvider, type LlmConfig } from "../../src/core/llm.js";

const VAR = "ORI_TEST_KEY_42_NEVER_SET";

function cfg(overrides: Partial<LlmConfig>): LlmConfig {
  return { provider: "anthropic", model: null, api_key_env: null, api_key_cmd: null, base_url: null, ...overrides };
}

describe("llmKeyProblem", () => {
  afterEach(() => { delete process.env[VAR]; });

  it("is silent when no provider is configured (a legitimate choice)", () => {
    expect(llmKeyProblem(cfg({ provider: null, api_key_env: VAR }))).toBeNull();
  });

  it("names the missing variable when api_key_env does not resolve", () => {
    const msg = llmKeyProblem(cfg({ api_key_env: VAR }));
    expect(msg).toContain(`"${VAR}"`);
    expect(msg).toContain("not present in the environment");
  });

  it("is silent once the variable is set", () => {
    process.env[VAR] = "sk-test";
    expect(llmKeyProblem(cfg({ api_key_env: VAR }))).toBeNull();
  });

  it("flags a provider with no key source at all", () => {
    expect(llmKeyProblem(cfg({}))).toContain("neither llm.api_key_env nor llm.api_key_cmd");
  });
});

describe("createProvider with an unresolvable key", () => {
  it("still falls back to NullProvider, but says why on stderr, once", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const c = cfg({ api_key_env: VAR });
      expect(await createProvider(c)).toBeInstanceOf(NullProvider);
      expect(await createProvider(c)).toBeInstanceOf(NullProvider);
      const hits = spy.mock.calls.filter((a) => String(a[0]).includes(VAR));
      expect(hits).toHaveLength(1);
    } finally {
      spy.mockRestore();
    }
  });
});

describe("ori health reports a configured-but-unusable LLM", () => {
  let vault: string;
  beforeEach(async () => {
    vault = await fs.mkdtemp(path.join(os.tmpdir(), "ori-llmhealth-"));
    await runInit({ targetDir: vault });
  });
  afterEach(async () => {
    delete process.env[VAR];
    await fs.rm(vault, { recursive: true, force: true });
  });

  async function setLlm(llm: Record<string, unknown>): Promise<void> {
    const file = path.join(vault, "ori.config.yaml");
    const doc = (YAML.parse(await fs.readFile(file, "utf8")) ?? {}) as Record<string, unknown>;
    doc.llm = llm;
    await fs.writeFile(file, YAML.stringify(doc), "utf8");
  }

  it("warns, naming the variable, when api_key_env is unset", async () => {
    await setLlm({ provider: "anthropic", api_key_env: VAR });
    const result = await runHealth(vault);
    expect(result.warnings.some((w) => w.includes(VAR) && w.includes("LLM features are disabled"))).toBe(true);
  });

  it("stays quiet when the variable is present", async () => {
    await setLlm({ provider: "anthropic", api_key_env: VAR });
    process.env[VAR] = "sk-test";
    const result = await runHealth(vault);
    expect(result.warnings.some((w) => w.includes("LLM features are disabled"))).toBe(false);
  });

  it("stays quiet when no LLM is configured", async () => {
    const result = await runHealth(vault);
    expect(result.warnings.some((w) => w.includes("LLM features are disabled"))).toBe(false);
  });
});
