import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DEFAULT_RERANKER_MODEL_PATH,
  defaultConfig,
  defaultConfigPath,
  listenBase,
  loadConfig,
  modelBase,
  rerankerBase,
  upstreamBase,
  validateConfig,
} from "../src/config.ts";

const temps: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "gbrain-embed-config-"));
  temps.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of temps.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("defaultConfig", () => {
  test("keeps the embedding defaults", () => {
    const cfg = defaultConfig();
    expect(cfg.listenPort).toBe(8082);
    expect(cfg.upstreamPort).toBe(8083);
    expect(cfg.nativeDimensions).toBe(2560);
    expect(cfg.dimensions).toBe(1536);
    expect(cfg.defaultMode).toBe("background");
    expect(cfg.modelId).toBe("qwen3-embedding-4b");
    expect(cfg.modelPath.endsWith("qwen3-embedding-4b-q6_k.gguf")).toBe(true);
  });

  test("adds the reranker defaults", () => {
    const cfg = defaultConfig();
    expect(cfg.reranker.enabled).toBe(true);
    expect(cfg.reranker.modelId).toBe("qwen3-reranker-0.6b");
    expect(cfg.reranker.upstreamPort).toBe(8084);
    expect(cfg.reranker.ttlSeconds).toBe(300);
    expect(cfg.reranker.taskpolicy).toBe(true);
    expect(cfg.reranker.modelPath).toBe(DEFAULT_RERANKER_MODEL_PATH);
    expect(cfg.reranker.modelPath.endsWith("qwen3-reranker-0.6b-q8_0.gguf")).toBe(true);
  });

  test("returns fresh mode and reranker objects on every call", () => {
    const first = defaultConfig();
    first.modes.background.ttlSeconds = 1;
    first.reranker.ttlSeconds = 1;
    const second = defaultConfig();
    expect(second.modes.background.ttlSeconds).toBe(120);
    expect(second.reranker.ttlSeconds).toBe(300);
  });
});

describe("loadConfig", () => {
  test("returns the defaults when no file is configured", () => {
    const cfg = loadConfig(join(tempDir(), "missing.json"));
    expect(cfg.listenPort).toBe(8082);
    expect(cfg.upstreamPort).toBe(8083);
    expect(cfg.dimensions).toBe(1536);
    expect(cfg.nativeDimensions).toBe(2560);
    expect(cfg.defaultMode).toBe("background");
    expect(cfg.reranker.upstreamPort).toBe(8084);
    expect(cfg.reranker.ttlSeconds).toBe(300);
  });

  test("defaults the config path to the user config directory", () => {
    expect(defaultConfigPath()).toContain(join("gbrain-embed", "config.json"));
  });

  test("merges a partial file over the defaults, including one mode profile", () => {
    const dir = tempDir();
    const path = join(dir, "config.json");
    writeFileSync(
      path,
      JSON.stringify({
        dimensions: 1024,
        defaultMode: "throughput",
        modes: { background: { ttlSeconds: 5 } },
      }),
    );
    const cfg = loadConfig(path);
    expect(cfg.dimensions).toBe(1024);
    expect(cfg.defaultMode).toBe("throughput");
    expect(cfg.modes.background.ttlSeconds).toBe(5);
    expect(cfg.modes.background.llamaArgs).toEqual(defaultConfig().modes.background.llamaArgs);
    expect(cfg.modes.throughput.ttlSeconds).toBe(1800);
  });

  test("merges a partial reranker object over its defaults", () => {
    const dir = tempDir();
    const path = join(dir, "config.json");
    writeFileSync(path, JSON.stringify({ reranker: { ttlSeconds: 900 } }));
    const cfg = loadConfig(path);
    expect(cfg.reranker.ttlSeconds).toBe(900);
    expect(cfg.reranker.enabled).toBe(true);
    expect(cfg.reranker.modelId).toBe("qwen3-reranker-0.6b");
    expect(cfg.reranker.upstreamPort).toBe(8084);
    expect(cfg.reranker.modelPath).toBe(DEFAULT_RERANKER_MODEL_PATH);
    expect(cfg.reranker.llamaArgs).toEqual(defaultConfig().reranker.llamaArgs);
  });

  const invalidReranker: Array<[string, Record<string, unknown>, RegExp]> = [
    ["a non-positive port", { upstreamPort: 0 }, /invalid reranker\.upstreamPort/],
    ["a port above 65535", { upstreamPort: 70_000 }, /invalid reranker\.upstreamPort/],
    ["a negative ttl", { ttlSeconds: -1 }, /invalid reranker\.ttlSeconds/],
    ["an empty modelPath", { modelPath: "" }, /invalid reranker\.modelPath/],
    ["an empty modelId", { modelId: "" }, /invalid reranker\.modelId/],
  ];

  for (const [name, patch, pattern] of invalidReranker) {
    test(`rejects ${name} in the config file`, () => {
      const dir = tempDir();
      const path = join(dir, "config.json");
      writeFileSync(path, JSON.stringify({ reranker: patch }));
      expect(() => loadConfig(path)).toThrow(pattern);
    });
  }

  // Fail closed on a malformed reranker field instead of spreading it over the
  // defaults; only an absent field (or a JSON object) is accepted.
  const nonObjectReranker: Array<[string, unknown]> = [
    ["a number", 42],
    ["null", null],
    ["an array", []],
    ["a string", "ab"],
  ];

  for (const [name, value] of nonObjectReranker) {
    test(`rejects ${name} in the reranker field`, () => {
      const dir = tempDir();
      const path = join(dir, "config.json");
      writeFileSync(path, JSON.stringify({ reranker: value }));
      expect(() => loadConfig(path)).toThrow(/invalid reranker config: expected an object/);
    });
  }
});

describe("validateConfig", () => {
  test("accepts the defaults", () => {
    expect(() => validateConfig(defaultConfig())).not.toThrow();
  });

  test("rejects equal listen and upstream ports", () => {
    const cfg = defaultConfig();
    cfg.upstreamPort = cfg.listenPort;
    expect(() => validateConfig(cfg)).toThrow(/must differ/);
  });

  test("rejects a width above the native width", () => {
    const cfg = defaultConfig();
    cfg.dimensions = 4096;
    expect(() => validateConfig(cfg)).toThrow(/below dimensions/);
  });

  test("rejects a listen port shared with the reranker", () => {
    const cfg = defaultConfig();
    cfg.reranker.upstreamPort = cfg.listenPort;
    expect(() => validateConfig(cfg)).toThrow(/listenPort and reranker\.upstreamPort must differ/);
  });

  test("rejects an embedding port shared with the reranker", () => {
    const cfg = defaultConfig();
    cfg.reranker.upstreamPort = cfg.upstreamPort;
    expect(() => validateConfig(cfg)).toThrow(/upstreamPort and reranker\.upstreamPort must differ/);
  });

  test("rejects an invalid reranker port", () => {
    const cfg = defaultConfig();
    cfg.reranker.upstreamPort = 70_000;
    expect(() => validateConfig(cfg)).toThrow(/invalid reranker\.upstreamPort 70000/);
  });

  test("rejects a negative reranker ttl", () => {
    const cfg = defaultConfig();
    cfg.reranker.ttlSeconds = -1;
    expect(() => validateConfig(cfg)).toThrow(/invalid reranker\.ttlSeconds -1/);
  });

  test("rejects an empty reranker modelPath", () => {
    const cfg = defaultConfig();
    cfg.reranker.modelPath = "";
    expect(() => validateConfig(cfg)).toThrow(/invalid reranker\.modelPath/);
  });

  test("rejects an empty reranker modelId", () => {
    const cfg = defaultConfig();
    cfg.reranker.modelId = "";
    expect(() => validateConfig(cfg)).toThrow(/invalid reranker\.modelId/);
  });
});

describe("URL helpers", () => {
  test("render loopback bases for both children", () => {
    const cfg = defaultConfig();
    expect(listenBase(cfg)).toBe("http://127.0.0.1:8082");
    expect(upstreamBase(cfg)).toBe("http://127.0.0.1:8083");
    expect(rerankerBase(cfg)).toBe("http://127.0.0.1:8084");
  });

  test("modelBase dispatches on the model kind", () => {
    const cfg = defaultConfig();
    expect(modelBase(cfg, "embedding")).toBe(upstreamBase(cfg));
    expect(modelBase(cfg, "reranker")).toBe(rerankerBase(cfg));
  });

  test("follows a reconfigured reranker port", () => {
    const cfg = defaultConfig();
    cfg.reranker.upstreamPort = 9099;
    expect(rerankerBase(cfg)).toBe("http://127.0.0.1:9099");
    expect(modelBase(cfg, "reranker")).toBe("http://127.0.0.1:9099");
  });
});
