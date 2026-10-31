/**
 * Configuration for the gbrain embedding supervisor.
 *
 * By default the service reads `~/.config/gbrain-embed/config.json` (or
 * `$XDG_CONFIG_HOME/gbrain-embed/config.json`); `GBRAIN_EMBED_CONFIG` or the
 * `--config` flag override it. Every field has a `$HOME`-based default so the
 * CLI and the tests can run before any config file exists.
 *
 * One proxy fronts two independently managed `llama-server` children: the
 * embedding model (the original single-model layout) and the reranker
 * (Qwen3-Reranker). They have separate ports, idle TTLs, logs and state
 * records, so either model can load and unload without touching the other.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type Mode = "background" | "throughput";

export interface ModeProfile {
  /** Seconds without an embedding request before the model server is stopped. */
  ttlSeconds: number;
  /** Wrap the child in `taskpolicy -b` (background QoS / throttled I/O). */
  taskpolicy: boolean;
  /** llama.cpp flags after the shared `-m/--alias/--embeddings/--pooling` set. */
  llamaArgs: string[];
}

/**
 * The reranker child. It is deliberately not part of a `ModeProfile`: the
 * background/throughput profiles size the embedding server's context and
 * batching, while the reranker has its own fixed launch profile and its own
 * idle TTL. Switching embedding modes never stops the reranker.
 */
export interface RerankerConfig {
  /** Serve `/v1/rerank`; when false the route answers 503 and setup leaves gbrain alone. */
  enabled: boolean;
  modelPath: string;
  modelId: string;
  upstreamPort: number;
  /** Seconds without a rerank request before the reranker server is stopped. */
  ttlSeconds: number;
  /** Wrap the child in `taskpolicy -b` (background QoS / throttled I/O). */
  taskpolicy: boolean;
  /** llama.cpp flags after the shared `-m/--alias/--reranking/--pooling` set. */
  llamaArgs: string[];
}

export interface EmbedConfig {
  listenHost: string;
  listenPort: number;
  upstreamHost: string;
  /** Port of the managed embedding `llama-server`. */
  upstreamPort: number;
  llamaServer: string;
  gbrainBin: string;
  modelPath: string;
  modelId: string;
  nativeDimensions: number;
  dimensions: number;
  defaultMode: Mode;
  modes: Record<Mode, ModeProfile>;
  reranker: RerankerConfig;
  stateDir: string;
  /** How often the idle reaper runs. */
  idleCheckMs: number;
  startupTimeoutMs: number;
  requestTimeoutMs: number;
  maxBodyBytes: number;
}

const home = homedir();

export const DEFAULT_MODEL_PATH = join(
  home,
  ".local/share/gbrain-embed/models/qwen3-embedding-4b-q6_k.gguf",
);
export const DEFAULT_RERANKER_MODEL_PATH = join(
  home,
  ".local/share/gbrain-embed/models/qwen3-reranker-0.6b-q8_0.gguf",
);
export const DEFAULT_STATE_DIR = join(home, ".local/state/gbrain-embed");
export const DEFAULT_LLAMA_SERVER = "/opt/homebrew/bin/llama-server";

function defaultGbrainBin(): string {
  const nixProfile = join(home, ".nix-profile/bin/gbrain");
  return existsSync(nixProfile) ? nixProfile : "gbrain";
}

export function defaultRerankerConfig(): RerankerConfig {
  return {
    enabled: true,
    modelPath: DEFAULT_RERANKER_MODEL_PATH,
    modelId: "qwen3-reranker-0.6b",
    upstreamPort: 8084,
    ttlSeconds: 300,
    taskpolicy: true,
    // Qwen3-Reranker is a 0.6B cross-encoder: a small context is enough for
    // the capped query+passage pairs. `--pooling rank` selects the classifier
    // head explicitly.
    llamaArgs: ["-c", "4096", "-b", "2048", "-ub", "2048", "--parallel", "1", "-t", "4", "-fa", "on"],
  };
}

export function defaultConfig(): EmbedConfig {
  return {
    listenHost: "127.0.0.1",
    listenPort: 8082,
    upstreamHost: "127.0.0.1",
    upstreamPort: 8083,
    llamaServer: DEFAULT_LLAMA_SERVER,
    gbrainBin: defaultGbrainBin(),
    modelPath: DEFAULT_MODEL_PATH,
    modelId: "qwen3-embedding-4b",
    nativeDimensions: 2560,
    dimensions: 1536,
    defaultMode: "background",
    modes: {
      background: {
        ttlSeconds: 120,
        taskpolicy: true,
        llamaArgs: ["-c", "2048", "-b", "512", "-ub", "512", "--parallel", "1", "-t", "2", "-fa", "on"],
      },
      throughput: {
        ttlSeconds: 1800,
        taskpolicy: false,
        llamaArgs: ["-c", "8192", "-b", "2048", "-ub", "2048", "--parallel", "2", "-t", "8", "-fa", "on"],
      },
    },
    reranker: defaultRerankerConfig(),
    stateDir: DEFAULT_STATE_DIR,
    idleCheckMs: 5_000,
    startupTimeoutMs: 90_000,
    requestTimeoutMs: 300_000,
    maxBodyBytes: 32 * 1024 * 1024,
  };
}

export function isMode(value: unknown): value is Mode {
  return value === "background" || value === "throughput";
}

function validatePort(name: string, value: unknown): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0 || (value as number) > 65535) {
    throw new Error(`invalid ${name} ${String(value)}`);
  }
}

export function validateConfig(cfg: EmbedConfig): EmbedConfig {
  if (!isMode(cfg.defaultMode)) throw new Error(`invalid defaultMode ${JSON.stringify(cfg.defaultMode)}`);
  for (const mode of ["background", "throughput"] as const) {
    const profile = cfg.modes[mode];
    if (!profile) throw new Error(`missing mode profile ${mode}`);
    if (!Number.isFinite(profile.ttlSeconds) || profile.ttlSeconds < 0) {
      throw new Error(`invalid ${mode}.ttlSeconds ${profile.ttlSeconds}`);
    }
    if (!Array.isArray(profile.llamaArgs)) throw new Error(`invalid ${mode}.llamaArgs`);
  }
  if (!Number.isInteger(cfg.dimensions) || cfg.dimensions <= 0) {
    throw new Error(`invalid dimensions ${cfg.dimensions}`);
  }
  if (!Number.isInteger(cfg.nativeDimensions) || cfg.nativeDimensions < cfg.dimensions) {
    throw new Error(`nativeDimensions ${cfg.nativeDimensions} is below dimensions ${cfg.dimensions}`);
  }
  if (!Number.isInteger(cfg.listenPort) || cfg.listenPort < 0 || cfg.listenPort > 65535) {
    throw new Error(`invalid listenPort ${cfg.listenPort}`);
  }
  validatePort("upstreamPort", cfg.upstreamPort);
  if (cfg.listenPort !== 0 && cfg.listenPort === cfg.upstreamPort) {
    throw new Error("listenPort and upstreamPort must differ");
  }

  const reranker = cfg.reranker;
  if (!reranker || typeof reranker !== "object") throw new Error("missing reranker config");
  if (typeof reranker.enabled !== "boolean") throw new Error("invalid reranker.enabled");
  if (typeof reranker.modelPath !== "string" || reranker.modelPath.length === 0) {
    throw new Error("invalid reranker.modelPath");
  }
  if (typeof reranker.modelId !== "string" || reranker.modelId.length === 0) {
    throw new Error("invalid reranker.modelId");
  }
  validatePort("reranker.upstreamPort", reranker.upstreamPort);
  if (!Number.isFinite(reranker.ttlSeconds) || reranker.ttlSeconds < 0) {
    throw new Error(`invalid reranker.ttlSeconds ${reranker.ttlSeconds}`);
  }
  if (!Array.isArray(reranker.llamaArgs)) throw new Error("invalid reranker.llamaArgs");
  if (cfg.listenPort !== 0 && cfg.listenPort === reranker.upstreamPort) {
    throw new Error("listenPort and reranker.upstreamPort must differ");
  }
  if (cfg.upstreamPort === reranker.upstreamPort) {
    throw new Error("upstreamPort and reranker.upstreamPort must differ");
  }
  return cfg;
}

/**
 * The config file the service reads when neither `GBRAIN_EMBED_CONFIG` nor
 * `--config` names one.
 */
export function defaultConfigPath(): string {
  const base = process.env.XDG_CONFIG_HOME ?? join(homedir(), ".config");
  return join(base, "gbrain-embed", "config.json");
}

/**
 * Load the configured file, or the defaults when it is absent. A missing
 * config is not an error: the CLI is useful before any config file exists.
 */
export function loadConfig(
  path: string | undefined = process.env.GBRAIN_EMBED_CONFIG ?? defaultConfigPath(),
): EmbedConfig {
  const base = defaultConfig();
  if (!path || !existsSync(path)) return validateConfig(base);
  const raw = JSON.parse(readFileSync(path, "utf8")) as Partial<EmbedConfig>;
  const modes = { ...base.modes };
  for (const mode of ["background", "throughput"] as const) {
    if (raw.modes?.[mode]) modes[mode] = { ...base.modes[mode], ...raw.modes[mode] };
  }
  const rawReranker: unknown = raw.reranker;
  if (
    rawReranker !== undefined &&
    (rawReranker === null || typeof rawReranker !== "object" || Array.isArray(rawReranker))
  ) {
    throw new Error("invalid reranker config: expected an object");
  }
  const reranker = {
    ...base.reranker,
    ...((rawReranker as Partial<RerankerConfig> | undefined) ?? {}),
  };
  return validateConfig({ ...base, ...raw, modes, reranker });
}

export function modeStatePath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "mode");
}

export function upstreamStatePath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "upstream.json");
}

export function upstreamOutLogPath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "llama-server.out.log");
}

export function upstreamErrLogPath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "llama-server.err.log");
}

export function rerankerStatePath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "reranker.json");
}

export function rerankerOutLogPath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "llama-reranker.out.log");
}

export function rerankerErrLogPath(cfg: EmbedConfig): string {
  return join(cfg.stateDir, "llama-reranker.err.log");
}

export function listenBase(cfg: EmbedConfig): string {
  return `http://${cfg.listenHost}:${cfg.listenPort}`;
}

export function upstreamBase(cfg: EmbedConfig): string {
  return `http://${cfg.upstreamHost}:${cfg.upstreamPort}`;
}

export function rerankerBase(cfg: EmbedConfig): string {
  return `http://${cfg.upstreamHost}:${cfg.reranker.upstreamPort}`;
}

/** The `http://host:port` base of either managed child. */
export function modelBase(cfg: EmbedConfig, kind: "embedding" | "reranker"): string {
  return kind === "reranker" ? rerankerBase(cfg) : upstreamBase(cfg);
}
