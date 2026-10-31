import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config.ts";
import { buildLlamaArgs, commandLineMatches } from "../src/upstream.ts";

describe("buildLlamaArgs", () => {
  test("background profile keeps the shared embedding flags and the mode flags", () => {
    const cfg = defaultConfig();
    const args = buildLlamaArgs(cfg, "background");
    expect(args.slice(0, 4)).toEqual([
      "-m", cfg.modelPath,
      "--alias", cfg.modelId,
    ]);
    expect(args).toContain("--embeddings");
    expect(args).toContain("--pooling");
    expect(args).toContain("last");
    expect(args).toContain("--no-webui");
    expect(args).toContain("2048");
    expect(args).toContain("2");
    expect(args[args.indexOf("--port") + 1]).toBe(String(cfg.upstreamPort));
  });

  test("throughput profile uses its own flags", () => {
    const cfg = defaultConfig();
    const args = buildLlamaArgs(cfg, "throughput");
    expect(args).toContain("8192");
    expect(args).toContain("8");
    expect(args[args.indexOf("--parallel") + 1]).toBe("2");
  });
});

describe("commandLineMatches", () => {
  const cfg = defaultConfig();

  test("matches our llama-server child", () => {
    const command = `/opt/homebrew/bin/llama-server -m ${cfg.modelPath} --port ${cfg.upstreamPort}`;
    expect(commandLineMatches(command, cfg)).toBe(true);
  });

  test("does not match another model or port", () => {
    expect(commandLineMatches("/opt/homebrew/bin/llama-server -m /tmp/other.gguf --port 9999", cfg)).toBe(false);
    expect(commandLineMatches(`/usr/bin/something -m ${cfg.modelPath}`, cfg)).toBe(false);
  });
});
