import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig } from "../src/config.ts";
import {
  installModel,
  parseOllamaTag,
  resolveOllamaBlob,
  sha256File,
} from "../src/model.ts";

const temps: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "gbrain-embed-model-"));
  temps.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of temps.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("parseOllamaTag", () => {
  test("splits a namespaced tag", () => {
    expect(parseOllamaTag("batiai/qwen3-embedding:4b")).toEqual({
      namespace: "batiai",
      name: "qwen3-embedding",
      tag: "4b",
    });
  });

  test("defaults a library tag to latest", () => {
    expect(parseOllamaTag("nomic-embed-text")).toEqual({
      namespace: "library",
      name: "nomic-embed-text",
      tag: "latest",
    });
  });
});

describe("resolveOllamaBlob", () => {
  test("resolves the model layer of a manifest", () => {
    const home = tempDir();
    const manifestDir = join(home, "models/manifests/registry.ollama.ai/batiai/qwen3-embedding");
    mkdirSync(manifestDir, { recursive: true });
    writeFileSync(
      join(manifestDir, "4b"),
      JSON.stringify({
        layers: [
          { mediaType: "application/vnd.ollama.image.template", digest: "sha256:aaaa" },
          { mediaType: "application/vnd.ollama.image.model", digest: "sha256:bbbb" },
        ],
      }),
    );
    mkdirSync(join(home, "models/blobs"), { recursive: true });
    writeFileSync(join(home, "models/blobs/sha256-bbbb"), "model");
    expect(resolveOllamaBlob("batiai/qwen3-embedding:4b", home)).toBe(
      join(home, "models/blobs/sha256-bbbb"),
    );
  });

  test("returns null when the manifest or blob is absent", () => {
    const home = tempDir();
    expect(resolveOllamaBlob("batiai/qwen3-embedding:4b", home)).toBeNull();
  });
});

describe("sha256File", () => {
  test("hashes the file's bytes", async () => {
    const dir = tempDir();
    const path = join(dir, "abc");
    writeFileSync(path, "abc");
    expect(await sha256File(path)).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("installModel", () => {
  test("refuses to overwrite a mismatched model without --refresh", async () => {
    const dir = tempDir();
    const target = join(dir, "model.gguf");
    writeFileSync(target, "not the model");
    const cfg = defaultConfig();
    cfg.modelPath = target;
    await expect(installModel(cfg)).rejects.toThrow(/SHA-256/);
  });

  test("removes a copied source whose digest does not match", async () => {
    const dir = tempDir();
    const source = join(dir, "source.gguf");
    const target = join(dir, "model.gguf");
    writeFileSync(source, "also not the model");
    const cfg = defaultConfig();
    cfg.modelPath = target;
    await expect(installModel(cfg, { source: `file:${source}` })).rejects.toThrow(/SHA-256/);
  });
});
