import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config.ts";
import { buildLlamaArgs, commandLineMatches, Upstream } from "../src/upstream.ts";

/** The argument after a flag, or undefined when the flag is absent. */
function flagValue(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag);
  return index === -1 ? undefined : args[index + 1];
}

describe("buildLlamaArgs", () => {
  test("background profile keeps the shared embedding flags and the mode flags", () => {
    const cfg = defaultConfig();
    const args = buildLlamaArgs(cfg, "background");
    expect(args.slice(0, 4)).toEqual(["-m", cfg.modelPath, "--alias", cfg.modelId]);
    expect(args).toContain("--embeddings");
    expect(flagValue(args, "--pooling")).toBe("last");
    expect(args).toContain("--no-webui");
    expect(flagValue(args, "-c")).toBe("2048");
    expect(flagValue(args, "--parallel")).toBe("1");
    expect(flagValue(args, "--host")).toBe(cfg.upstreamHost);
    expect(flagValue(args, "--port")).toBe(String(cfg.upstreamPort));
    expect(args).not.toContain("--reranking");
    expect(args).not.toContain(cfg.reranker.modelPath);
  });

  test("throughput profile uses its own flags", () => {
    const cfg = defaultConfig();
    const args = buildLlamaArgs(cfg, "throughput");
    expect(flagValue(args, "-c")).toBe("8192");
    expect(flagValue(args, "-t")).toBe("8");
    expect(flagValue(args, "--parallel")).toBe("2");
    expect(args).toContain("--embeddings");
  });

  test("reranker profile uses its own model, port and flags", () => {
    const cfg = defaultConfig();
    const args = buildLlamaArgs(cfg, "background", "reranker");
    expect(args.slice(0, 4)).toEqual(["-m", cfg.reranker.modelPath, "--alias", cfg.reranker.modelId]);
    expect(args).toContain("--reranking");
    expect(flagValue(args, "--pooling")).toBe("rank");
    expect(flagValue(args, "-c")).toBe("4096");
    expect(flagValue(args, "-b")).toBe("2048");
    expect(flagValue(args, "-ub")).toBe("2048");
    expect(flagValue(args, "--parallel")).toBe("1");
    expect(flagValue(args, "--host")).toBe(cfg.upstreamHost);
    expect(flagValue(args, "--port")).toBe(String(cfg.reranker.upstreamPort));
    expect(args).toContain("--no-webui");
    expect(args).toEqual(expect.arrayContaining(cfg.reranker.llamaArgs));
    // The two launch modes are mutually exclusive in llama.cpp.
    expect(args).not.toContain("--embeddings");
    expect(args).not.toContain(cfg.modelPath);
  });

  test("the reranker profile does not change with the embedding mode", () => {
    const cfg = defaultConfig();
    expect(buildLlamaArgs(cfg, "throughput", "reranker")).toEqual(
      buildLlamaArgs(cfg, "background", "reranker"),
    );
  });
});

describe("commandLineMatches", () => {
  const cfg = defaultConfig();
  const embeddingCommand = `/opt/homebrew/bin/llama-server -m ${cfg.modelPath} --port ${cfg.upstreamPort}`;
  const rerankerCommand =
    `/opt/homebrew/bin/llama-server -m ${cfg.reranker.modelPath} --port ${cfg.reranker.upstreamPort}`;

  test("matches the embedding child only for the embedding kind", () => {
    expect(commandLineMatches(embeddingCommand, cfg)).toBe(true);
    expect(commandLineMatches(embeddingCommand, cfg, "embedding")).toBe(true);
    expect(commandLineMatches(embeddingCommand, cfg, "reranker")).toBe(false);
  });

  test("matches the reranker child only for the reranker kind", () => {
    expect(commandLineMatches(rerankerCommand, cfg, "reranker")).toBe(true);
    expect(commandLineMatches(rerankerCommand, cfg)).toBe(false);
    expect(commandLineMatches(rerankerCommand, cfg, "embedding")).toBe(false);
  });

  test("does not match another model or a non-llama process", () => {
    expect(
      commandLineMatches("/opt/homebrew/bin/llama-server -m /tmp/other.gguf --port 9999", cfg),
    ).toBe(false);
    expect(commandLineMatches(`/usr/bin/something -m ${cfg.modelPath}`, cfg)).toBe(false);
  });
});

describe("Upstream status", () => {
  test("the reranker child reports its own kind before it starts", () => {
    const upstream = new Upstream(defaultConfig(), "reranker");
    const status = upstream.status();
    expect(status.kind).toBe("reranker");
    expect(status.state).toBe("stopped");
    expect(status.pid).toBeUndefined();
    expect(upstream.isRunning()).toBe(false);
  });

  test("the embedding child is the default kind", () => {
    const status = new Upstream(defaultConfig()).status();
    expect(status.kind).toBe("embedding");
    expect(status.state).toBe("stopped");
  });
});
