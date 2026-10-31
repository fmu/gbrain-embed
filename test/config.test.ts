import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  defaultConfig,
  defaultConfigPath,
  listenBase,
  loadConfig,
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

describe("loadConfig", () => {
  test("returns the defaults when no file is configured", () => {
    const cfg = loadConfig(join(tempDir(), "missing.json"));
    expect(cfg.listenPort).toBe(8082);
    expect(cfg.upstreamPort).toBe(8083);
    expect(cfg.dimensions).toBe(1536);
    expect(cfg.nativeDimensions).toBe(2560);
    expect(cfg.defaultMode).toBe("background");
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
});

describe("validateConfig", () => {
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
});

describe("URL helpers", () => {
  test("render loopback bases", () => {
    const cfg = defaultConfig();
    expect(listenBase(cfg)).toBe("http://127.0.0.1:8082");
    expect(upstreamBase(cfg)).toBe("http://127.0.0.1:8083");
  });
});
