/**
 * The loopback OpenAI-compatible endpoint gbrain talks to.
 *
 * It fronts two managed `llama-server` children and owns the lifecycle gbrain's
 * `llama-server` recipes cannot: start each server on the first request that
 * needs it, and stop it after its own idle window. The embedding and reranker
 * children are independent — separate ports, idle clocks, logs and PID records
 * — so an idle reranker unloads without touching a busy embedder (and vice
 * versa), and an embedding `mode` switch never stops the reranker.
 *
 * Embeddings are fitted to the brain's vector width with the Matryoshka prefix
 * and a renormalization; rerank requests are proxied to llama.cpp's
 * `POST /v1/rerank` (`{results: [{index, relevance_score}]}`), which is exactly
 * what the `llama-server-reranker` gbrain recipe expects.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { EmbedConfig, Mode } from "./config.ts";
import { isMode, modeStatePath, rerankerBase, upstreamBase } from "./config.ts";
import { encodeEmbeddingBase64, truncateAndNormalize } from "./truncate.ts";
import type { UpstreamLike, UpstreamStatus } from "./upstream.ts";

/** One model child plus the idle bookkeeping the supervisor keeps for it. */
export interface ModelStatus {
  upstream: UpstreamStatus;
  lastRequestAt: number | null;
  idleTtlMs: number;
  modelPath: string;
  modelId: string;
  upstreamPort: number;
  rssBytes: number | null;
}

export interface RerankerStatus extends ModelStatus {
  enabled: boolean;
}

export interface ServiceStatus {
  ok: true;
  mode: Mode;
  /** The embedding child (kept at this key for older `/health` consumers). */
  upstream: UpstreamStatus;
  lastRequestAt: number | null;
  idleTtlMs: number;
  modelPath: string;
  modelId: string;
  dimensions: number;
  nativeDimensions: number;
  listenPort: number;
  upstreamPort: number;
  rssBytes: number | null;
  /** The reranker child, with its own independent idle TTL. */
  reranker: RerankerStatus;
}

export type UnloadTarget = "embedding" | "reranker" | "all";

interface JsonErrorBody {
  error: { message: string; type: string; code?: string };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function errorBody(message: string, type: string): JsonErrorBody {
  return { error: { message, type } };
}

function isUnloadTarget(value: unknown): value is UnloadTarget {
  return value === "embedding" || value === "reranker" || value === "all";
}

export function readMode(cfg: EmbedConfig): Mode {
  try {
    const path = modeStatePath(cfg);
    if (existsSync(path)) {
      const value = readFileSync(path, "utf8").trim();
      if (isMode(value)) return value;
    }
  } catch {
    /* fall through to the configured default */
  }
  return cfg.defaultMode;
}

export function writeMode(cfg: EmbedConfig, mode: Mode): void {
  mkdirSync(cfg.stateDir, { recursive: true });
  writeFileSync(modeStatePath(cfg), `${mode}\n`, { mode: 0o600 });
}

function rssOf(upstream: UpstreamLike): number | null {
  return typeof (upstream as { rssBytes?: () => number | null }).rssBytes === "function"
    ? (upstream as { rssBytes: () => number | null }).rssBytes()
    : null;
}

function recoverOrphan(upstream: UpstreamLike): void {
  if (typeof (upstream as { recoverOrphan?: () => void }).recoverOrphan === "function") {
    (upstream as { recoverOrphan: () => void }).recoverOrphan();
  }
}

export class EmbedService {
  private server: ReturnType<typeof Bun.serve> | null = null;
  private idleTimer: ReturnType<typeof setInterval> | null = null;
  private mode: Mode;
  private lastEmbeddingRequestAt: number | null = null;
  private lastRerankRequestAt: number | null = null;
  private reaping = false;
  /** Requests currently talking to a child; the idle reaper must never unload one mid-flight. */
  private inFlight = { embedding: 0, reranker: 0 };
  /** A mode switch arrived while an embedding request was in flight. */
  private pendingEmbeddingStop = false;
  private embeddingStart: Promise<void> | null = null;
  private rerankerStart: Promise<void> | null = null;

  constructor(
    private readonly cfg: EmbedConfig,
    private readonly embedding: UpstreamLike,
    private readonly reranker: UpstreamLike,
  ) {
    this.mode = readMode(cfg);
  }

  start(): ReturnType<typeof Bun.serve> {
    if (this.server) return this.server;
    this.embedding.status();
    this.reranker.status();
    recoverOrphan(this.embedding);
    recoverOrphan(this.reranker);
    this.server = Bun.serve({
      hostname: this.cfg.listenHost,
      port: this.cfg.listenPort,
      fetch: (request) => this.handle(request),
    });
    this.idleTimer = setInterval(() => {
      void this.reap();
    }, this.cfg.idleCheckMs);
    return this.server;
  }

  get port(): number {
    return this.server?.port ?? this.cfg.listenPort;
  }

  get currentMode(): Mode {
    return this.mode;
  }

  async stop(): Promise<void> {
    if (this.idleTimer) {
      clearInterval(this.idleTimer);
      this.idleTimer = null;
    }
    if (this.server) {
      this.server.stop(true);
      this.server = null;
    }
    await Promise.all([this.embedding.stop(), this.reranker.stop()]);
  }

  status(): ServiceStatus {
    const embedding = this.embedding.status();
    const reranker = this.reranker.status();
    return {
      ok: true,
      mode: this.mode,
      upstream: embedding,
      lastRequestAt: this.lastEmbeddingRequestAt,
      idleTtlMs: this.cfg.modes[this.mode].ttlSeconds * 1000,
      modelPath: this.cfg.modelPath,
      modelId: this.cfg.modelId,
      dimensions: this.cfg.dimensions,
      nativeDimensions: this.cfg.nativeDimensions,
      listenPort: this.port,
      upstreamPort: this.cfg.upstreamPort,
      rssBytes: rssOf(this.embedding),
      reranker: {
        enabled: this.cfg.reranker.enabled,
        upstream: reranker,
        lastRequestAt: this.lastRerankRequestAt,
        idleTtlMs: this.cfg.reranker.ttlSeconds * 1000,
        modelPath: this.cfg.reranker.modelPath,
        modelId: this.cfg.reranker.modelId,
        upstreamPort: this.cfg.reranker.upstreamPort,
        rssBytes: rssOf(this.reranker),
      },
    };
  }

  /**
   * Switch the embedding resource profile. The embedding child is stopped so
   * the next embedding request starts it with the new flags; the reranker is
   * deliberately left alone because its profile does not depend on the mode.
   */
  async setMode(mode: Mode): Promise<void> {
    writeMode(this.cfg, mode);
    this.mode = mode;
    // Do not yank the child out from under an in-flight request; stop it as
    // soon as that request finishes. A child that is still cold-starting has
    // already captured the old mode, so it is stopped after the request too.
    // The reranker is untouched either way.
    const state = this.embedding.status().state;
    if (state === "running" && this.inFlight.embedding === 0) {
      await this.embedding.stop();
    } else if (state !== "stopped") {
      this.pendingEmbeddingStop = true;
    }
  }

  /**
   * Mark a request as in flight for one child. The idle reaper skips a child
   * with a request in flight, which matters during a cold start: the model can
   * take longer to load than the idle TTL, and unloading it then would kill
   * the very request that woke it.
   */
  private async track<T>(kind: "embedding" | "reranker", fn: () => Promise<T>): Promise<T> {
    this.inFlight[kind] += 1;
    try {
      return await fn();
    } finally {
      this.inFlight[kind] -= 1;
      // Count idle time from completion, not from arrival.
      if (kind === "embedding") this.lastEmbeddingRequestAt = Date.now();
      else this.lastRerankRequestAt = Date.now();
      if (kind === "embedding" && this.pendingEmbeddingStop && this.inFlight.embedding === 0) {
        this.pendingEmbeddingStop = false;
        await this.embedding.stop();
      }
    }
  }

  private async reap(): Promise<void> {
    if (this.reaping) return;
    this.reaping = true;
    try {
      if (this.inFlight.embedding === 0) {
        await this.reapOne(
          this.embedding,
          this.lastEmbeddingRequestAt,
          this.cfg.modes[this.mode].ttlSeconds,
        );
      }
      if (this.cfg.reranker.enabled && this.inFlight.reranker === 0) {
        await this.reapOne(this.reranker, this.lastRerankRequestAt, this.cfg.reranker.ttlSeconds);
      }
    } finally {
      this.reaping = false;
    }
  }

  private async reapOne(
    upstream: UpstreamLike,
    lastRequestAt: number | null,
    ttlSeconds: number,
  ): Promise<void> {
    if (!upstream.isRunning()) return;
    const startedAt = upstream.status().startedAt ?? Date.now();
    const since = lastRequestAt ?? startedAt;
    if (ttlSeconds >= 0 && Date.now() - since > ttlSeconds * 1000) {
      await upstream.stop();
    }
  }

  /** Concurrent cold embedding requests share one spawn. */
  private ensureEmbedding(): Promise<void> {
    if (this.embedding.isRunning()) return Promise.resolve();
    if (!this.embeddingStart) {
      this.embeddingStart = this.embedding.start(this.mode).finally(() => {
        this.embeddingStart = null;
      });
    }
    return this.embeddingStart;
  }

  /** Concurrent cold rerank requests share one spawn, independently of the embedding child. */
  private ensureReranker(): Promise<void> {
    if (!this.cfg.reranker.enabled) {
      return Promise.reject(new Error("the reranker is disabled in the config"));
    }
    if (this.reranker.isRunning()) return Promise.resolve();
    if (!this.rerankerStart) {
      this.rerankerStart = this.reranker.start(this.mode).finally(() => {
        this.rerankerStart = null;
      });
    }
    return this.rerankerStart;
  }

  private async unload(target: UnloadTarget): Promise<void> {
    const embedding = target === "all" || target === "embedding";
    const reranker = target === "all" || target === "reranker";
    const stops: Promise<void>[] = [];
    if (embedding) stops.push(this.embedding.stop());
    if (reranker) stops.push(this.reranker.stop());
    await Promise.all(stops);
  }

  private async handle(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    try {
      if (path === "/health" && request.method === "GET") {
        return json(this.status());
      }
      if (path === "/v1/models" && request.method === "GET") {
        const data = [
          {
            id: this.cfg.modelId,
            object: "model",
            created: 0,
            owned_by: "gbrain-embed",
          },
        ];
        if (this.cfg.reranker.enabled) {
          data.push({
            id: this.cfg.reranker.modelId,
            object: "model",
            created: 0,
            owned_by: "gbrain-embed",
          });
        }
        return json({ object: "list", data });
      }
      if (path === "/v1/embeddings" && request.method === "POST") {
        return await this.handleEmbeddings(request);
      }
      if ((path === "/v1/rerank" || path === "/rerank") && request.method === "POST") {
        return await this.handleRerank(request);
      }
      if (path === "/admin/mode" && request.method === "POST") {
        const body = await this.readJson(request);
        const mode = (body as { mode?: unknown }).mode;
        if (!isMode(mode)) {
          return json(errorBody("mode must be \"background\" or \"throughput\"", "invalid_request_error"), 400);
        }
        await this.setMode(mode);
        return json(this.status());
      }
      if (path === "/admin/unload" && request.method === "POST") {
        let target: UnloadTarget = "all";
        const text = await request.text();
        if (text.trim()) {
          let body: unknown;
          try {
            body = JSON.parse(text);
          } catch {
            return json(errorBody("request body is not valid JSON", "invalid_request_error"), 400);
          }
          const requested = (body as { model?: unknown } | null)?.model;
          if (requested !== undefined) {
            if (!isUnloadTarget(requested)) {
              return json(
                errorBody('model must be "embedding", "reranker" or "all"', "invalid_request_error"),
                400,
              );
            }
            target = requested;
          }
        }
        await this.unload(target);
        return json(this.status());
      }
      if (path === "/admin/restart" && request.method === "POST") {
        // Backwards-compatible alias for "drop both children"; the next
        // request restarts the one it needs.
        await this.unload("all");
        return json(this.status());
      }
      return json(errorBody(`no route for ${request.method} ${path}`, "invalid_request_error"), 404);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return json(errorBody(message, "internal_error"), 500);
    }
  }

  private async readJson(request: Request): Promise<unknown> {
    const text = await request.text();
    if (Buffer.byteLength(text, "utf8") > this.cfg.maxBodyBytes) {
      throw new Error(`request body exceeds ${this.cfg.maxBodyBytes} bytes`);
    }
    try {
      return JSON.parse(text);
    } catch {
      throw new Error("request body is not valid JSON");
    }
  }

  private async handleEmbeddings(request: Request): Promise<Response> {
    let body: Record<string, unknown>;
    try {
      body = (await this.readJson(request)) as Record<string, unknown>;
    } catch (error) {
      return json(errorBody(error instanceof Error ? error.message : String(error), "invalid_request_error"), 400);
    }

    const input = body.input;
    const count = typeof input === "string" ? 1 : Array.isArray(input) ? input.length : 0;
    if (count === 0) {
      return json(errorBody("input must be a string or a non-empty array of strings", "invalid_request_error"), 400);
    }
    const requested = Number(body.dimensions);
    const dimensions =
      Number.isInteger(requested) && requested > 0 ? requested : this.cfg.dimensions;
    if (dimensions > this.cfg.nativeDimensions) {
      return json(
        errorBody(
          `dimensions ${dimensions} exceeds the model's native ${this.cfg.nativeDimensions}`,
          "invalid_request_error",
        ),
        400,
      );
    }
    const wantsBase64 = body.encoding_format === "base64";

    this.lastEmbeddingRequestAt = Date.now();
    return this.track("embedding", () => this.forwardEmbedding(body, count, dimensions, wantsBase64));
  }

  private async forwardEmbedding(
    body: Record<string, unknown>,
    count: number,
    dimensions: number,
    wantsBase64: boolean,
  ): Promise<Response> {
    try {
      await this.ensureEmbedding();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return json(errorBody(`embedding server unavailable: ${message}`, "upstream_unavailable"), 503);
    }

    const forwarded = { ...body, encoding_format: "float" };
    let response: Response;
    try {
      response = await fetch(`${upstreamBase(this.cfg)}/v1/embeddings`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(forwarded),
        signal: AbortSignal.timeout(this.cfg.requestTimeoutMs),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await this.embedding.stop();
      return json(errorBody(`embedding server request failed: ${message}`, "upstream_error"), 502);
    }

    const text = await response.text();
    if (!response.ok) {
      return new Response(text, {
        status: response.status,
        headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
      });
    }

    let parsed: { data?: Array<{ embedding?: unknown; index?: number }>; [key: string]: unknown };
    try {
      parsed = JSON.parse(text);
    } catch {
      return json(errorBody("embedding server returned invalid JSON", "upstream_error"), 502);
    }
    if (!Array.isArray(parsed.data) || parsed.data.length !== count) {
      return json(
        errorBody(
          `embedding server returned ${parsed.data?.length ?? 0} vectors for ${count} inputs`,
          "upstream_error",
        ),
        502,
      );
    }
    try {
      for (const item of parsed.data) {
        if (!Array.isArray(item.embedding)) {
          throw new Error("embedding server returned a non-array embedding");
        }
        const fitted = truncateAndNormalize(item.embedding as number[], dimensions);
        item.embedding = wantsBase64 ? encodeEmbeddingBase64(fitted) : fitted;
      }
    } catch (error) {
      return json(errorBody(error instanceof Error ? error.message : String(error), "upstream_error"), 502);
    }

    return json(parsed);
  }

  private async handleRerank(request: Request): Promise<Response> {
    if (!this.cfg.reranker.enabled) {
      return json(
        errorBody(
          "the reranker is disabled; set `reranker.enabled` to true in the config",
          "reranker_disabled",
        ),
        503,
      );
    }
    let body: Record<string, unknown>;
    try {
      body = (await this.readJson(request)) as Record<string, unknown>;
    } catch (error) {
      return json(errorBody(error instanceof Error ? error.message : String(error), "invalid_request_error"), 400);
    }

    const query = body.query;
    const documents = body.documents;
    if (typeof query !== "string" || query.length === 0) {
      return json(errorBody("query must be a non-empty string", "invalid_request_error"), 400);
    }
    if (
      !Array.isArray(documents) ||
      documents.length === 0 ||
      documents.some((document) => typeof document !== "string")
    ) {
      return json(
        errorBody("documents must be a non-empty array of strings", "invalid_request_error"),
        400,
      );
    }

    this.lastRerankRequestAt = Date.now();
    return this.track("reranker", () => this.forwardRerank(body));
  }

  private async forwardRerank(body: Record<string, unknown>): Promise<Response> {
    try {
      await this.ensureReranker();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return json(errorBody(`reranker server unavailable: ${message}`, "upstream_unavailable"), 503);
    }

    // The proxy owns one reranker alias, so the request is pinned to it; the
    // rest of the caller's body (query, documents, top_n, ...) is forwarded.
    const forwarded = { ...body, model: this.cfg.reranker.modelId };
    let response: Response;
    try {
      response = await fetch(`${rerankerBase(this.cfg)}/v1/rerank`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(forwarded),
        signal: AbortSignal.timeout(this.cfg.requestTimeoutMs),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await this.reranker.stop();
      return json(errorBody(`reranker request failed: ${message}`, "upstream_error"), 502);
    }

    const text = await response.text();
    return new Response(text, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  }
}
