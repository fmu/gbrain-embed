import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig, type EmbedConfig } from "../src/config.ts";
import {
  DEFAULT_HF_FILE,
  DEFAULT_HF_REPO,
  DEFAULT_OLLAMA_TAG,
  EMBEDDING_MODEL_SHA256,
  EMBEDDING_MODEL_SIZE_BYTES,
  RERANKER_HF_FILE,
  RERANKER_HF_REPO,
  RERANKER_MODEL_SHA256,
  RERANKER_MODEL_SIZE_BYTES,
  installModel,
  installModels,
  modelDescriptors,
  parseOllamaTag,
  resolveOllamaBlob,
  sha256File,
} from "../src/model.ts";

const temps: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "qwen3-embed-model-"));
  temps.push(dir);
  return dir;
}

/** A config whose two model targets live inside a fresh temp directory. */
function localConfig(): { cfg: EmbedConfig; dir: string } {
  const dir = tempDir();
  const cfg = defaultConfig();
  cfg.modelPath = join(dir, "embedding.gguf");
  cfg.reranker.modelPath = join(dir, "reranker.gguf");
  return { cfg, dir };
}

afterEach(() => {
  for (const dir of temps.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("modelDescriptors", () => {
  test("describes both models and keeps the embedding Ollama tag", () => {
    const cfg = defaultConfig();
    const descriptors = modelDescriptors(cfg);
    expect(Object.keys(descriptors)).toEqual(["embedding", "reranker"]);

    const embedding = descriptors.embedding;
    expect(embedding.key).toBe("embedding");
    expect(embedding.path).toBe(cfg.modelPath);
    expect(embedding.modelId).toBe(cfg.modelId);
    expect(embedding.sha256).toBe(EMBEDDING_MODEL_SHA256);
    expect(embedding.sizeBytes).toBe(EMBEDDING_MODEL_SIZE_BYTES);
    expect(embedding.hfRepo).toBe(DEFAULT_HF_REPO);
    expect(embedding.hfFile).toBe(DEFAULT_HF_FILE);
    expect(embedding.ollamaTag).toBe(DEFAULT_OLLAMA_TAG);

    const reranker = descriptors.reranker;
    expect(reranker.key).toBe("reranker");
    expect(reranker.path).toBe(cfg.reranker.modelPath);
    expect(reranker.modelId).toBe(cfg.reranker.modelId);
    expect(reranker.sha256).toBe(RERANKER_MODEL_SHA256);
    expect(reranker.sizeBytes).toBe(RERANKER_MODEL_SIZE_BYTES);
    expect(reranker.hfRepo).toBe(RERANKER_HF_REPO);
    expect(reranker.hfFile).toBe(RERANKER_HF_FILE);
    // Only the embedding bytes come from an Ollama layer; the reranker is HF-only.
    expect(reranker.ollamaTag).toBeUndefined();
  });

  test("reflects shadowed paths and ids", () => {
    const cfg = defaultConfig();
    cfg.modelPath = "/tmp/embedding.gguf";
    cfg.modelId = "custom-embedding";
    cfg.reranker.modelPath = "/tmp/reranker.gguf";
    cfg.reranker.modelId = "custom-reranker";
    const { embedding, reranker } = modelDescriptors(cfg);
    expect(embedding.path).toBe("/tmp/embedding.gguf");
    expect(embedding.modelId).toBe("custom-embedding");
    expect(reranker.path).toBe("/tmp/reranker.gguf");
    expect(reranker.modelId).toBe("custom-reranker");
    // The pinned digests stay pinned regardless of path or alias.
    expect(embedding.sha256).toBe(EMBEDDING_MODEL_SHA256);
    expect(reranker.sha256).toBe(RERANKER_MODEL_SHA256);
  });
});

/**
 * A matching-digest success needs the real multi-GB GGUF, so the offline suite
 * can only pin the rejection, cleanup and ordering behaviour. The success path
 * would return `source: "existing"` without touching the disk.
 */
describe("installModel", () => {
  test("refuses to overwrite a mismatched embedding file without --refresh", async () => {
    const { cfg } = localConfig();
    writeFileSync(cfg.modelPath, "not the model");
    await expect(installModel(cfg, "embedding")).rejects.toThrow(/SHA-256/);
    expect(existsSync(cfg.modelPath)).toBe(true);
  });

  test("refuses to overwrite a mismatched reranker file without --refresh", async () => {
    const { cfg } = localConfig();
    writeFileSync(cfg.reranker.modelPath, "not the reranker");
    await expect(installModel(cfg, "reranker")).rejects.toThrow(/SHA-256/);
    expect(existsSync(cfg.reranker.modelPath)).toBe(true);
  });

  test("removes a copied embedding source whose digest does not match", async () => {
    const { cfg, dir } = localConfig();
    const source = join(dir, "source.gguf");
    writeFileSync(source, "also not the model");
    await expect(installModel(cfg, "embedding", { source: `file:${source}` })).rejects.toThrow(
      /SHA-256/,
    );
    expect(existsSync(cfg.modelPath)).toBe(false);
  });

  test("removes a copied reranker source whose digest does not match", async () => {
    const { cfg, dir } = localConfig();
    const source = join(dir, "source.gguf");
    writeFileSync(source, "also not the reranker");
    await expect(installModel(cfg, "reranker", { source: `file:${source}` })).rejects.toThrow(
      /SHA-256/,
    );
    expect(existsSync(cfg.reranker.modelPath)).toBe(false);
  });

  test("rejects a missing embedding source", async () => {
    const { cfg, dir } = localConfig();
    const missing = join(dir, "absent.gguf");
    await expect(installModel(cfg, "embedding", { source: `file:${missing}` })).rejects.toThrow(
      /does not exist/,
    );
    expect(existsSync(cfg.modelPath)).toBe(false);
  });

  test("rejects a missing reranker source", async () => {
    const { cfg, dir } = localConfig();
    const missing = join(dir, "absent.gguf");
    await expect(installModel(cfg, "reranker", { source: `file:${missing}` })).rejects.toThrow(
      /does not exist/,
    );
    expect(existsSync(cfg.reranker.modelPath)).toBe(false);
  });

  test("treats a bare path as a file source", async () => {
    const { cfg, dir } = localConfig();
    const source = join(dir, "source.gguf");
    writeFileSync(source, "still not the reranker");
    await expect(installModel(cfg, "reranker", { source })).rejects.toThrow(/SHA-256/);
    expect(existsSync(cfg.reranker.modelPath)).toBe(false);
  });

  test("--refresh skips the existing-file check but still verifies the source", async () => {
    const { cfg, dir } = localConfig();
    writeFileSync(cfg.modelPath, "not the model");
    const missing = join(dir, "absent.gguf");
    await expect(
      installModel(cfg, "embedding", { refresh: true, source: `file:${missing}` }),
    ).rejects.toThrow(/does not exist/);
    // The copy failed before replacing the target, so the old file is intact.
    expect(existsSync(cfg.modelPath)).toBe(true);
  });
});

describe("installModels", () => {
  test("stops after the first failing key", async () => {
    const { cfg, dir } = localConfig();
    writeFileSync(cfg.modelPath, "not the model");
    const missing = join(dir, "absent.gguf");
    const error = await installModels(cfg, ["embedding", "reranker"], {
      source: `file:${missing}`,
    }).catch((reason: unknown) => reason);
    // The embedding failure (mismatched existing file) wins; reaching the
    // reranker would have reported the missing source instead.
    expect(String(error)).toMatch(/SHA-256/);
    expect(String(error)).not.toMatch(/does not exist/);
    expect(existsSync(cfg.reranker.modelPath)).toBe(false);
  });
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
