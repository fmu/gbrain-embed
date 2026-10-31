import { describe, expect, test } from "bun:test";
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
});
