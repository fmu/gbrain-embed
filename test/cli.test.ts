import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.ts";

async function captureHelp(): Promise<string> {
  const chunks: string[] = [];
  const original = process.stdout.write;
  process.stdout.write = ((chunk: string | Uint8Array) => {
    chunks.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8"));
    return true;
  }) as typeof process.stdout.write;
  try {
    await main(["--help"]);
  } finally {
    process.stdout.write = original;
  }
  return chunks.join("");
}

describe("cli", () => {
  test("exports the command entry point", () => {
    expect(typeof main).toBe("function");
  });

  test("help documents both models and the per-model commands", async () => {
    const help = await captureHelp();
    expect(help).toContain("reranker");
    expect(help).toContain("qwen3-reranker-0.6b");
    expect(help).toContain("model install [embedding|reranker|all]");
    expect(help).toContain("independent of the embedding mode");
  });

  test("help separates model setup from the optional gbrain integration", async () => {
    const help = await captureHelp();
    expect(help).toContain("setup [--json]");
    expect(help).toContain("setup-gbrain [--json]");
  });

  test("setup-gbrain refuses before touching gbrain when models are missing", async () => {
    const dir = mkdtempSync(join(tmpdir(), "qwen3-embed-cli-"));
    const configPath = join(dir, "config.json");
    writeFileSync(
      configPath,
      JSON.stringify({
        modelPath: join(dir, "missing-embedding.gguf"),
        reranker: { enabled: false },
      }),
    );
    const previous = process.env.QWEN3_EMBED_CONFIG;
    try {
      await expect(main(["setup-gbrain", "--config", configPath, "--json"])).rejects.toThrow(
        /models are not installed/,
      );
    } finally {
      if (previous === undefined) delete process.env.QWEN3_EMBED_CONFIG;
      else process.env.QWEN3_EMBED_CONFIG = previous;
    }
  });
});
