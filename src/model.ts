/**
 * The model files: the embedding GGUF and the reranker GGUF, each verified by
 * SHA-256 and kept outside the Nix store because they are large and change
 * only when the model does.
 *
 * The embedding bytes are the same file Ollama pulls for
 * `batiai/qwen3-embedding:4b` (the Ollama manifest's model layer digest matches
 * the HF LFS object), so the installer hardlinks the already-present blob when
 * it can and falls back to a resumable Hugging Face download when it cannot.
 *
 * The reranker is the llama.cpp maintainers' Qwen3-Reranker-0.6B conversion
 * (`ggml-org`), which carries the classifier head (`cls.output.weight`). Some
 * community conversions drop that tensor: the model then loads and answers,
 * but every score is meaningless. The SHA-256 pin rejects them.
 */
import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  linkSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { EmbedConfig } from "./config.ts";

export type ModelKey = "embedding" | "reranker";

export const EMBEDDING_MODEL_SHA256 =
  "7c94391728b84154fe79b4c4e69f6a0adb4314f6ff24811f55db1ae88461d99f";
export const EMBEDDING_MODEL_SIZE_BYTES = 3_305_684_512;
export const DEFAULT_OLLAMA_TAG = "batiai/qwen3-embedding:4b";
export const DEFAULT_HF_REPO = "batiai/Qwen3-Embedding-4B-GGUF";
export const DEFAULT_HF_FILE = "Qwen3-Embedding-4B-Q6_K.gguf";

/** Qwen3-Reranker-0.6B Q8_0, converted by the llama.cpp maintainers. */
export const RERANKER_MODEL_SHA256 =
  "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48";
export const RERANKER_MODEL_SIZE_BYTES = 639_153_184;
export const RERANKER_HF_REPO = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF";
export const RERANKER_HF_FILE = "qwen3-reranker-0.6b-q8_0.gguf";

/** Legacy names kept for the embedding-only call sites. */
export const MODEL_SHA256 = EMBEDDING_MODEL_SHA256;
export const MODEL_SIZE_BYTES = EMBEDDING_MODEL_SIZE_BYTES;

export interface ModelDescriptor {
  key: ModelKey;
  /** Short human label used by the CLI. */
  label: string;
  modelId: string;
  path: string;
  sha256: string;
  sizeBytes: number;
  hfRepo: string;
  hfFile: string;
  /** Ollama tag whose model layer is byte-identical, when one exists. */
  ollamaTag?: string;
}

export interface InstallOptions {
  refresh?: boolean;
  /** `ollama:<tag>`, `hf:<repo>/<file>`, or a filesystem path. */
  source?: string;
}

export interface InstallResult {
  key: ModelKey;
  path: string;
  sha256: string;
  source: string;
  bytes: number;
}

export function modelDescriptors(cfg: EmbedConfig): Record<ModelKey, ModelDescriptor> {
  return {
    embedding: {
      key: "embedding",
      label: "embedding",
      modelId: cfg.modelId,
      path: cfg.modelPath,
      sha256: EMBEDDING_MODEL_SHA256,
      sizeBytes: EMBEDDING_MODEL_SIZE_BYTES,
      hfRepo: DEFAULT_HF_REPO,
      hfFile: DEFAULT_HF_FILE,
      ollamaTag: DEFAULT_OLLAMA_TAG,
    },
    reranker: {
      key: "reranker",
      label: "reranker",
      modelId: cfg.reranker.modelId,
      path: cfg.reranker.modelPath,
      sha256: RERANKER_MODEL_SHA256,
      sizeBytes: RERANKER_MODEL_SIZE_BYTES,
      hfRepo: RERANKER_HF_REPO,
      hfFile: RERANKER_HF_FILE,
    },
  };
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  const stream = Bun.file(path).stream();
  for await (const chunk of stream) hash.update(chunk as Uint8Array);
  return hash.digest("hex");
}

export function parseOllamaTag(tag: string): { namespace: string; name: string; tag: string } {
  const colon = tag.lastIndexOf(":");
  const namePart = colon === -1 ? tag : tag.slice(0, colon);
  const tagPart = colon === -1 ? "latest" : tag.slice(colon + 1);
  const slash = namePart.indexOf("/");
  const namespace = slash === -1 ? "library" : namePart.slice(0, slash);
  const name = slash === -1 ? namePart : namePart.slice(slash + 1);
  return { namespace, name, tag: tagPart };
}

/** Resolve an Ollama tag to the GGUF blob it points at, or null when absent. */
export function resolveOllamaBlob(
  tag: string = DEFAULT_OLLAMA_TAG,
  ollamaHome: string = join(homedir(), ".ollama"),
): string | null {
  const parsed = parseOllamaTag(tag);
  const manifest = join(
    ollamaHome,
    "models/manifests/registry.ollama.ai",
    parsed.namespace,
    parsed.name,
    parsed.tag,
  );
  if (!existsSync(manifest)) return null;
  let layers: Array<{ mediaType?: string; digest?: string }>;
  try {
    layers = (JSON.parse(readFileSync(manifest, "utf8")) as { layers?: typeof layers }).layers ?? [];
  } catch {
    return null;
  }
  const model = layers.find((layer) => layer.mediaType === "application/vnd.ollama.image.model");
  if (!model?.digest?.startsWith("sha256:")) return null;
  const blob = join(ollamaHome, "models/blobs", `sha256-${model.digest.slice("sha256:".length)}`);
  return existsSync(blob) ? blob : null;
}

function hfUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}

function parseSource(source: string): { kind: "file" | "ollama" | "hf"; value: string } {
  if (source.startsWith("file:")) return { kind: "file", value: source.slice("file:".length) };
  if (source.startsWith("ollama:")) return { kind: "ollama", value: source.slice("ollama:".length) };
  if (source.startsWith("hf:")) return { kind: "hf", value: source.slice("hf:".length) };
  return { kind: "file", value: source };
}

function linkOrCopy(source: string, target: string): void {
  mkdirSync(dirname(target), { recursive: true });
  rmSync(target, { force: true });
  try {
    linkSync(source, target);
  } catch {
    copyFileSync(source, target);
  }
}

async function download(url: string, target: string): Promise<void> {
  mkdirSync(dirname(target), { recursive: true });
  const part = `${target}.part`;
  const child = Bun.spawn({
    cmd: ["curl", "--fail", "--location", "--continue-at", "-", "--output", part, url],
    stdin: "ignore",
    stdout: "inherit",
    stderr: "inherit",
  });
  const code = await child.exited;
  if (code !== 0) {
    throw new Error(`curl exited with ${code} while downloading ${url}`);
  }
  renameSync(part, target);
}

/**
 * Install one model at its configured path and verify it. Idempotent: an
 * existing file with the expected digest is returned without touching the
 * disk.
 */
export async function installModel(
  cfg: EmbedConfig,
  key: ModelKey,
  options: InstallOptions = {},
): Promise<InstallResult> {
  const descriptor = modelDescriptors(cfg)[key];
  const target = descriptor.path;
  if (existsSync(target) && !options.refresh) {
    const digest = await sha256File(target);
    if (digest === descriptor.sha256) {
      return {
        key,
        path: target,
        sha256: digest,
        source: "existing",
        bytes: statSync(target).size,
      };
    }
    throw new Error(
      `${target} exists but its SHA-256 is ${digest}, not ${descriptor.sha256}; ` +
        "re-run with --refresh to replace it",
    );
  }

  const fallbackSource = descriptor.ollamaTag
    ? `ollama:${descriptor.ollamaTag}`
    : `hf:${descriptor.hfRepo}/${descriptor.hfFile}`;
  const requested = options.source ?? fallbackSource;
  const parsed = parseSource(requested);
  let usedSource = requested;
  let downloaded = false;

  if (parsed.kind === "file") {
    if (!existsSync(parsed.value)) throw new Error(`model source does not exist: ${parsed.value}`);
    linkOrCopy(parsed.value, target);
  } else if (parsed.kind === "ollama") {
    const blob = resolveOllamaBlob(parsed.value);
    if (blob) {
      linkOrCopy(blob, target);
      usedSource = `ollama:${parsed.value} (${blob})`;
    } else {
      usedSource = `hf:${descriptor.hfRepo}/${descriptor.hfFile}`;
      await download(hfUrl(descriptor.hfRepo, descriptor.hfFile), target);
      downloaded = true;
    }
  } else {
    const slash = parsed.value.lastIndexOf("/");
    const repo = slash === -1 ? descriptor.hfRepo : parsed.value.slice(0, slash);
    const file = slash === -1 ? parsed.value : parsed.value.slice(slash + 1);
    await download(hfUrl(repo, file), target);
    downloaded = true;
  }

  const digest = await sha256File(target);
  if (digest !== descriptor.sha256) {
    rmSync(target, { force: true });
    throw new Error(
      `model at ${target} has SHA-256 ${digest}, expected ${descriptor.sha256}; the file was removed`,
    );
  }
  const bytes = statSync(target).size;
  if (!downloaded && bytes !== descriptor.sizeBytes) {
    // A hardlink/copy from an unexpected source is allowed only if the digest
    // matches, but a size mismatch means the digest check is the one that
    // matters; report the size for the operator.
    console.error(`[gbrain-embed] model size is ${bytes} bytes (expected ${descriptor.sizeBytes})`);
  }
  return { key, path: target, sha256: digest, source: usedSource, bytes };
}

/** Install several models in order; stops at the first failure. */
export async function installModels(
  cfg: EmbedConfig,
  keys: ModelKey[],
  options: InstallOptions = {},
): Promise<InstallResult[]> {
  const results: InstallResult[] = [];
  for (const key of keys) results.push(await installModel(cfg, key, options));
  return results;
}
