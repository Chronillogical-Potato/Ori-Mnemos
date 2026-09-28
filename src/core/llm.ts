import { execSync } from "node:child_process";

/* ------------------------------------------------------------------ */
/*  Chat interface (generic LLM calls for explore recursion)           */
/* ------------------------------------------------------------------ */

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatOptions {
  maxTokens?: number;
  temperature?: number;
}

/* ------------------------------------------------------------------ */
/*  Enhancement types (note promotion)                                 */
/* ------------------------------------------------------------------ */

export type VaultContext = {
  existingTitles: string[];
  recentNotes: Array<{ title: string; type: string; description: string }>;
  projectTags: string[];
};

export type EnhancementSuggestions = {
  type?: string;
  description?: string;
  project?: string[];
  reasoning?: string;
};

export type LlmConfig = {
  provider: string | null;
  model: string | null;
  api_key_env: string | null;
  api_key_cmd: string | null;
  base_url: string | null;
};

export const DEFAULT_LLM_CONFIG: LlmConfig = {
  provider: null,
  model: null,
  api_key_env: null,
  api_key_cmd: null,
  base_url: null,
};

export interface LlmProvider {
  enhance(
    note: {
      title: string;
      body: string;
      frontmatter: Record<string, unknown>;
    },
    context: VaultContext
  ): Promise<EnhancementSuggestions>;

  /** Generic chat completion for explore recursion and other internal reasoning. */
  chat(messages: ChatMessage[], options?: ChatOptions): Promise<string>;
}

/**
 * Null provider: returns empty suggestions (pure deterministic path).
 */
export class NullProvider implements LlmProvider {
  async enhance(): Promise<EnhancementSuggestions> {
    return {};
  }
  async chat(): Promise<string> {
    return "";
  }
}

function resolveApiKey(config: LlmConfig): string | undefined {
  if (config.api_key_cmd) {
    try {
      const key = execSync(config.api_key_cmd, {
        encoding: "utf-8",
        timeout: 5000,
        stdio: ["ignore", "pipe", "pipe"],
      }).trim();
      if (key) {
        return key;
      }
    } catch (err) {
      console.error(`api_key_cmd failed: ${(err as Error).message}`);
    }
  }

  if (config.api_key_env) {
    return process.env[config.api_key_env];
  }

  return undefined;
}

/**
 * Explain why a configured LLM provider has no usable API key, or return null
 * when there is nothing wrong that can be seen without running api_key_cmd.
 *
 * "Not configured" (provider: null) is a legitimate choice and returns null.
 * "Configured but unusable" is a configuration error: before #42 it fell back
 * to NullProvider silently, and the only visible symptom was a generic
 * "configure LLM" hint that blamed the user while notes piled up with empty
 * descriptions.
 */
export function llmKeyProblem(config: LlmConfig): string | null {
  if (!config.provider) return null;
  if (config.api_key_cmd) return null; // only knowable by running it; createProvider reports that case
  if (config.api_key_env) {
    // Same test resolveApiKey applies, so health and runtime agree.
    if (process.env[config.api_key_env]) return null;
    return (
      `llm.api_key_env "${config.api_key_env}" is set in ori.config.yaml, but ` +
      `${config.api_key_env} is not present in the environment — LLM features are disabled.`
    );
  }
  return (
    `llm.provider "${config.provider}" is set in ori.config.yaml, but neither ` +
    `llm.api_key_env nor llm.api_key_cmd is set — LLM features are disabled.`
  );
}

// One stderr line per distinct problem per process: createProvider runs once
// per promote/explore, and a long-lived `ori serve` should not repeat itself.
const reportedKeyProblems = new Set<string>();

function reportKeyProblem(message: string): void {
  if (reportedKeyProblems.has(message)) return;
  reportedKeyProblems.add(message);
  console.error(`[ori] ${message}`);
}

/**
 * Create provider from config. Returns NullProvider when provider is null.
 * A provider that is configured but has no resolvable key also yields
 * NullProvider, but says so on stderr (never stdout: CLI JSON and MCP stdio
 * both live there).
 */
export async function createProvider(config: LlmConfig): Promise<LlmProvider> {
  if (!config.provider) {
    return new NullProvider();
  }

  const apiKey = resolveApiKey(config);

  if (!apiKey) {
    reportKeyProblem(
      llmKeyProblem(config) ??
        `llm.api_key_cmd produced no key (and no llm.api_key_env fallback resolved) — LLM features are disabled.`,
    );
    return new NullProvider();
  }

  switch (config.provider) {
    case "anthropic": {
      const { AnthropicProvider } = await import("../providers/anthropic.js");
      return new AnthropicProvider(
        apiKey,
        config.model ?? "claude-sonnet-4-20250514"
      );
    }
    case "openai": {
      const { OpenAICompatProvider } = await import("../providers/openai-compat.js");
      return new OpenAICompatProvider(
        apiKey,
        config.model ?? "gpt-4o",
        config.base_url
      );
    }
    default:
      return new NullProvider();
  }
}
