/**
 * The loopback OpenAI-compatible endpoint gbrain talks to.
 *
 * It owns two things gbrain's `llama-server` recipe cannot: starting the model
 * server on the first embedding request, and stopping it after an idle window.
 * It also fits Qwen3-Embedding's 2560-wide output to the brain's 1536-wide
 * column by taking the Matryoshka prefix and renormalizing.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { EmbedConfig, Mode } from "./config.ts";
import { isMode, modeStatePath, upstreamBase } from "./config.ts";
import { encodeEmbeddingBase64, truncateAndNormalize } from "./truncate.ts";
import type { UpstreamLike, UpstreamStatus } from "./upstream.ts";

export interface ServiceStatus {
  ok: true;
  mode: Mode;
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
}

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

export class EmbedService {
  private server: ReturnType<typeof Bun.serve> | null = null;
  private idleTimer: ReturnType<typeof setInterval> | null = null;
  private mode: Mode;
  private lastRequestAt: number | null = null;
  private reaping = false;
  private startPromise: Promise<void> | null = null;

  constructor(
    private readonly cfg: EmbedConfig,
    private readonly upstream: UpstreamLike,
  ) {
    this.mode = readMode(cfg);
  }

  start(): ReturnType<typeof Bun.serve> {
    if (this.server) return this.server;
    this.upstream.status();
    if (typeof (this.upstream as { recoverOrphan?: () => void }).recoverOrphan === "function") {
      (this.upstream as { recoverOrphan: () => void }).recoverOrphan();
    }
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
    await this.upstream.stop();
  }

  status(): ServiceStatus {
    const upstream = this.upstream.status();
    const rssBytes =
      typeof (this.upstream as { rssBytes?: () => number | null }).rssBytes === "function"
        ? (this.upstream as { rssBytes: () => number | null }).rssBytes()
        : null;
    return {
      ok: true,
      mode: this.mode,
      upstream,
      lastRequestAt: this.lastRequestAt,
      idleTtlMs: this.cfg.modes[this.mode].ttlSeconds * 1000,
      modelPath: this.cfg.modelPath,
      modelId: this.cfg.modelId,
      dimensions: this.cfg.dimensions,
      nativeDimensions: this.cfg.nativeDimensions,
      listenPort: this.port,
      upstreamPort: this.cfg.upstreamPort,
      rssBytes,
    };
  }

  async setMode(mode: Mode): Promise<void> {
    writeMode(this.cfg, mode);
    this.mode = mode;
    if (this.upstream.isRunning()) await this.upstream.stop();
  }

  private async reap(): Promise<void> {
    if (this.reaping || !this.upstream.isRunning()) return;
    this.reaping = true;
    try {
      const startedAt = this.upstream.status().startedAt ?? Date.now();
      const since = this.lastRequestAt ?? startedAt;
      const ttlMs = this.cfg.modes[this.mode].ttlSeconds * 1000;
      if (ttlMs >= 0 && Date.now() - since > ttlMs) {
        await this.upstream.stop();
      }
    } finally {
      this.reaping = false;
    }
  }

  /** Concurrent cold requests share one spawn. */
  private ensureUpstream(): Promise<void> {
    if (this.upstream.isRunning()) return Promise.resolve();
    if (!this.startPromise) {
      this.startPromise = this.upstream.start(this.mode).finally(() => {
        this.startPromise = null;
      });
    }
    return this.startPromise;
  }

  private async handle(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    try {
      if (path === "/health" && request.method === "GET") {
        return json(this.status());
      }
      if (path === "/v1/models" && request.method === "GET") {
        return json({
          object: "list",
          data: [
            {
              id: this.cfg.modelId,
              object: "model",
              created: 0,
              owned_by: "gbrain-embed",
            },
          ],
        });
      }
      if (path === "/v1/embeddings" && request.method === "POST") {
        return await this.handleEmbeddings(request);
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
      if (path === "/admin/restart" && request.method === "POST") {
        await this.upstream.stop();
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

    this.lastRequestAt = Date.now();
    try {
      await this.ensureUpstream();
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
      await this.upstream.stop();
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

    this.lastRequestAt = Date.now();
    return json(parsed);
  }
}
