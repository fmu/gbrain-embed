import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig, type EmbedConfig, type Mode } from "../src/config.ts";
import { EmbedService } from "../src/service.ts";
import { decodeEmbeddingBase64 } from "../src/truncate.ts";
import type { UpstreamLike, UpstreamStatus } from "../src/upstream.ts";

/** A managed child that never spawns; records how often it was started and stopped. */
class FakeUpstream implements UpstreamLike {
  starts = 0;
  stops = 0;
  running = false;
  /** Set while `start()` is sleeping, mirroring llama.cpp's health-wait window. */
  starting = false;
  mode: Mode = "background";
  startedAt: number | null = null;
  startDelayMs = 0;

  constructor(private readonly kind: "embedding" | "reranker") {}

  async start(mode: Mode): Promise<void> {
    this.starts += 1;
    this.starting = true;
    if (this.startDelayMs > 0) await Bun.sleep(this.startDelayMs);
    this.starting = false;
    this.running = true;
    this.mode = mode;
    this.startedAt = Date.now();
  }

  async stop(): Promise<void> {
    this.stops += 1;
    this.starting = false;
    this.running = false;
    this.startedAt = null;
  }

  isRunning(): boolean {
    return this.running;
  }

  status(): UpstreamStatus {
    const state = this.running ? "running" : this.starting ? "starting" : "stopped";
    return {
      state,
      kind: this.kind,
      ...(this.running && this.startedAt !== null ? { startedAt: this.startedAt } : {}),
      mode: this.mode,
    };
  }
}

interface FakeRequest {
  path: string;
  body: Record<string, unknown>;
}

interface FakeHttp {
  server: ReturnType<typeof Bun.serve>;
  requests: FakeRequest[];
}

/** Embedding fake: one vector per input, in request order. */
function startEmbeddingServer(vectors: number[][], status = 200, responseDelayMs = 0): FakeHttp {
  const requests: FakeRequest[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname !== "/v1/embeddings") return new Response("not found", { status: 404 });
      const body = (await request.json()) as { input?: unknown };
      requests.push({ path: url.pathname, body });
      if (responseDelayMs > 0) await Bun.sleep(responseDelayMs);
      if (status !== 200) {
        return new Response(JSON.stringify({ error: { message: "embedding boom" } }), {
          status,
          headers: { "content-type": "application/json" },
        });
      }
      const count = Array.isArray(body.input) ? body.input.length : 1;
      const data = Array.from({ length: count }, (_, index) => ({
        object: "embedding",
        embedding: vectors[index % vectors.length],
        index,
      }));
      return new Response(
        JSON.stringify({ object: "list", data, model: "fake-embedding", usage: { prompt_tokens: count } }),
        { headers: { "content-type": "application/json" } },
      );
    },
  });
  return { server, requests };
}

const RERANK_RESULTS = [
  { index: 0, relevance_score: 0.9 },
  { index: 1, relevance_score: 0.2 },
];

/** Reranker fake: answers `{results: [{index, relevance_score}]}` for `/v1/rerank`. */
function startRerankerServer(status = 200, responseDelayMs = 0): FakeHttp {
  const requests: FakeRequest[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname !== "/v1/rerank") return new Response("not found", { status: 404 });
      const body = (await request.json()) as Record<string, unknown>;
      requests.push({ path: url.pathname, body });
      if (responseDelayMs > 0) await Bun.sleep(responseDelayMs);
      if (status !== 200) {
        return new Response(JSON.stringify({ error: { message: "rerank boom" } }), {
          status,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(JSON.stringify({ results: RERANK_RESULTS }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });
  return { server, requests };
}

/** A listener that accepts and immediately closes, so fetch fails like a dead upstream. */
function startClosedPort(): { port: number; stop: () => void } {
  const listener = Bun.listen<unknown>({
    hostname: "127.0.0.1",
    port: 0,
    socket: {
      open(socket) {
        socket.end();
      },
      data(socket) {
        socket.end();
      },
    },
  });
  return { port: listener.port, stop: () => listener.stop(true) };
}

interface MakeOptions {
  vectors?: number[][];
  embeddingStatus?: number;
  rerankStatus?: number;
  rerankerEnabled?: boolean;
  embeddingTtlSeconds?: number;
  rerankerTtlSeconds?: number;
  embeddingStartDelayMs?: number;
  rerankerStartDelayMs?: number;
  /** Delay the fake child's HTTP answer so a request stays in flight for a while. */
  embeddingResponseDelayMs?: number;
  rerankResponseDelayMs?: number;
  /** Point the reranker child at a closing listener instead of the fake server. */
  rerankerUnreachable?: boolean;
}

interface Harness {
  cfg: EmbedConfig;
  service: EmbedService;
  base: string;
  embedding: FakeUpstream;
  reranker: FakeUpstream;
  embeddingHttp: FakeHttp;
  rerankHttp: FakeHttp;
  stop: () => Promise<void>;
}

interface HealthBody {
  ok: boolean;
  mode: string;
  dimensions: number;
  modelId: string;
  lastRequestAt: number | null;
  idleTtlMs: number;
  upstream: UpstreamStatus;
  reranker: {
    enabled: boolean;
    upstream: UpstreamStatus;
    lastRequestAt: number | null;
    idleTtlMs: number;
    modelId: string;
  };
}

interface ModelsBody {
  object: string;
  data: Array<{ id: string; object: string }>;
}

interface RerankBody {
  results: Array<{ index: number; relevance_score: number }>;
}

interface ErrorBody {
  error: { message: string; type: string };
}

let harness: Harness | null = null;

function makeService(options: MakeOptions = {}): Harness {
  const temp = mkdtempSync(join(tmpdir(), "qwen3-embed-test-"));
  const embeddingHttp = startEmbeddingServer(
    options.vectors ?? [[3, 4, 0, 0]],
    options.embeddingStatus ?? 200,
    options.embeddingResponseDelayMs ?? 0,
  );
  const rerankHttp = startRerankerServer(
    options.rerankStatus ?? 200,
    options.rerankResponseDelayMs ?? 0,
  );
  const closedRerankPort = options.rerankerUnreachable ? startClosedPort() : null;

  const cfg = defaultConfig();
  cfg.listenHost = "127.0.0.1";
  cfg.listenPort = 0;
  cfg.stateDir = temp;
  cfg.modelPath = join(temp, "model.gguf");
  cfg.nativeDimensions = 4;
  cfg.dimensions = 2;
  cfg.idleCheckMs = 20;
  cfg.upstreamPort = embeddingHttp.server.port;
  cfg.modes.background.ttlSeconds = options.embeddingTtlSeconds ?? 60;
  cfg.modes.throughput.ttlSeconds = options.embeddingTtlSeconds ?? 60;
  cfg.reranker.enabled = options.rerankerEnabled ?? true;
  cfg.reranker.modelPath = join(temp, "reranker.gguf");
  cfg.reranker.upstreamPort = closedRerankPort?.port ?? rerankHttp.server.port;
  cfg.reranker.ttlSeconds = options.rerankerTtlSeconds ?? 60;

  const embedding = new FakeUpstream("embedding");
  embedding.startDelayMs = options.embeddingStartDelayMs ?? 0;
  const reranker = new FakeUpstream("reranker");
  reranker.startDelayMs = options.rerankerStartDelayMs ?? 0;

  const service = new EmbedService(cfg, embedding, reranker);
  service.start();

  const stop = async () => {
    await service.stop();
    await embeddingHttp.server.stop(true);
    await rerankHttp.server.stop(true);
    closedRerankPort?.stop();
    rmSync(temp, { recursive: true, force: true });
  };

  const current: Harness = {
    cfg,
    service,
    base: `http://127.0.0.1:${service.port}`,
    embedding,
    reranker,
    embeddingHttp,
    rerankHttp,
    stop,
  };
  harness = current;
  return current;
}

function embed(h: Harness, body: unknown): Promise<Response> {
  return fetch(`${h.base}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function rerank(h: Harness, body: unknown): Promise<Response> {
  return fetch(`${h.base}/v1/rerank`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function admin(h: Harness, path: string, body?: unknown): Promise<Response> {
  return fetch(`${h.base}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
}

async function health(h: Harness): Promise<HealthBody> {
  const response = await fetch(`${h.base}/health`);
  expect(response.status).toBe(200);
  return (await response.json()) as HealthBody;
}

async function models(h: Harness): Promise<ModelsBody> {
  const response = await fetch(`${h.base}/v1/models`);
  expect(response.status).toBe(200);
  return (await response.json()) as ModelsBody;
}

/** Poll until the fake has observed the request, so the assertion is not timing-based. */
async function waitFor(predicate: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitFor timed out");
    await Bun.sleep(5);
  }
}

afterEach(async () => {
  const current = harness;
  harness = null;
  if (current) await current.stop();
});

describe("EmbedService embeddings", () => {
  test("starts only the embedding child and fits the vector to the configured width", async () => {
    const h = makeService();
    const response = await embed(h, { input: ["a"], dimensions: 2 });
    expect(response.status).toBe(200);
    const body = (await response.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([0.6, 0.8]);
    expect(h.embedding.starts).toBe(1);
    expect(h.reranker.starts).toBe(0);
  });

  test("honours a request for the native width without truncating", async () => {
    const h = makeService();
    const response = await embed(h, { input: ["a"], dimensions: 4 });
    const body = (await response.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([3, 4, 0, 0]);
  });

  test("concurrent cold embedding requests start the embedding child exactly once", async () => {
    const h = makeService({ embeddingStartDelayMs: 50 });
    const responses = await Promise.all(
      Array.from({ length: 5 }, () => embed(h, { input: ["a"], dimensions: 2 })),
    );
    for (const response of responses) expect(response.status).toBe(200);
    expect(h.embedding.starts).toBe(1);
    expect(h.reranker.starts).toBe(0);
  });

  test("surfaces an embedding upstream error status and its body", async () => {
    const h = makeService({ embeddingStatus: 500 });
    const response = await embed(h, { input: ["a"], dimensions: 2 });
    expect(response.status).toBe(500);
    expect(await response.text()).toContain("embedding boom");
  });

  test("encodes base64 when the caller asks for it", async () => {
    const h = makeService();
    const response = await embed(h, { input: ["a"], dimensions: 2, encoding_format: "base64" });
    const body = (await response.json()) as { data: Array<{ embedding: string }> };
    expect(typeof body.data[0].embedding).toBe("string");
    const decoded = decodeEmbeddingBase64(body.data[0].embedding);
    expect(decoded[0]).toBeCloseTo(0.6, 6);
    expect(decoded[1]).toBeCloseTo(0.8, 6);
  });
});

describe("EmbedService rerank", () => {
  test("starts only the reranker child and proxies query/documents to /v1/rerank", async () => {
    const h = makeService();
    const response = await rerank(h, { query: "which is longer?", documents: ["a", "b"], top_n: 2 });
    expect(response.status).toBe(200);
    const body = (await response.json()) as RerankBody;
    expect(body.results).toEqual(RERANK_RESULTS);
    expect(h.reranker.starts).toBe(1);
    expect(h.embedding.starts).toBe(0);
    expect(h.rerankHttp.requests).toHaveLength(1);
    expect(h.rerankHttp.requests[0].path).toBe("/v1/rerank");
    expect(h.rerankHttp.requests[0].body).toMatchObject({
      query: "which is longer?",
      documents: ["a", "b"],
      top_n: 2,
      model: h.cfg.reranker.modelId,
    });
    expect(h.embeddingHttp.requests).toHaveLength(0);
  });

  test("concurrent cold rerank requests start the reranker exactly once", async () => {
    const h = makeService({ rerankerStartDelayMs: 50 });
    const responses = await Promise.all(
      Array.from({ length: 5 }, () => rerank(h, { query: "q", documents: ["a", "b"] })),
    );
    for (const response of responses) expect(response.status).toBe(200);
    expect(h.reranker.starts).toBe(1);
    expect(h.embedding.starts).toBe(0);
    expect(h.rerankHttp.requests).toHaveLength(5);
  });

  test("passes a rerank upstream error status and body through", async () => {
    const h = makeService({ rerankStatus: 500 });
    const response = await rerank(h, { query: "q", documents: ["a"] });
    expect(response.status).toBe(500);
    expect(await response.text()).toContain("rerank boom");
  });

  test("answers 502 and drops the child when the reranker is unreachable", async () => {
    const h = makeService({ rerankerUnreachable: true });
    const response = await rerank(h, { query: "q", documents: ["a"] });
    expect(response.status).toBe(502);
    const body = (await response.json()) as ErrorBody;
    expect(body.error.type).toBe("upstream_error");
    expect(body.error.message).toContain("reranker request failed");
    expect(h.reranker.starts).toBe(1);
    expect(h.reranker.running).toBe(false);
  });

  test("rejects invalid rerank payloads without starting the reranker", async () => {
    const h = makeService();
    const bad: unknown[] = [
      {},
      { query: "", documents: ["a"] },
      { query: 42, documents: ["a"] },
      { query: "q" },
      { query: "q", documents: [] },
      { query: "q", documents: ["a", 7] },
    ];
    for (const body of bad) {
      const response = await rerank(h, body);
      expect(response.status).toBe(400);
      const error = (await response.json()) as ErrorBody;
      expect(error.error.type).toBe("invalid_request_error");
    }
    const badJson = await fetch(`${h.base}/v1/rerank`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not json",
    });
    expect(badJson.status).toBe(400);
    expect(h.reranker.starts).toBe(0);
  });

  test("leaves /v1/embeddings untouched when the reranker is disabled", async () => {
    const h = makeService({ rerankerEnabled: false });
    const response = await rerank(h, { query: "q", documents: ["a"] });
    expect(response.status).toBe(503);
    const body = (await response.json()) as ErrorBody;
    expect(body.error.type).toBe("reranker_disabled");
    expect(body.error.message).toContain("disabled");
    expect(h.reranker.starts).toBe(0);

    const listing = await models(h);
    expect(listing.data.map((model) => model.id)).toEqual([h.cfg.modelId]);

    const status = await health(h);
    expect(status.reranker.enabled).toBe(false);

    const embeddingResponse = await embed(h, { input: ["a"], dimensions: 2 });
    expect(embeddingResponse.status).toBe(200);
    expect(h.embedding.starts).toBe(1);
  });
});

describe("EmbedService model lifecycle", () => {
  test("lists both model ids without starting either child", async () => {
    const h = makeService();
    const body = await models(h);
    expect(body.object).toBe("list");
    expect(body.data.map((model) => model.id)).toEqual([
      h.cfg.modelId,
      h.cfg.reranker.modelId,
    ]);
    expect(h.embedding.starts).toBe(0);
    expect(h.reranker.starts).toBe(0);
  });

  test("unloads the reranker on its own ttl while the embedding child keeps running", async () => {
    const h = makeService({ embeddingTtlSeconds: 60, rerankerTtlSeconds: 0.15 });
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);

    await Bun.sleep(400);
    expect(h.reranker.running).toBe(false);
    expect(h.embedding.running).toBe(true);
  });

  test("unloads the embedding child on its own ttl while the reranker keeps running", async () => {
    const h = makeService({ embeddingTtlSeconds: 0.15, rerankerTtlSeconds: 60 });
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);

    await Bun.sleep(400);
    expect(h.embedding.running).toBe(false);
    expect(h.reranker.running).toBe(true);
  });

  test("switching the embedding mode leaves the reranker running", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);

    const response = await admin(h, "/admin/mode", { mode: "throughput" });
    expect(response.status).toBe(200);
    expect(h.embedding.running).toBe(false);
    expect(h.reranker.running).toBe(true);
    expect(h.reranker.starts).toBe(1);

    await embed(h, { input: ["a"], dimensions: 2 });
    expect(h.embedding.starts).toBe(2);
    expect(h.embedding.mode).toBe("throughput");
    expect(h.reranker.mode).toBe("background");
  });

  test("rejects an unknown mode without touching either child", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/mode", { mode: "turbo" });
    expect(response.status).toBe(400);
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);
    expect(h.service.currentMode).toBe("background");
  });

  test("unload embedding stops only the embedding child", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/unload", { model: "embedding" });
    expect(response.status).toBe(200);
    expect(h.embedding.running).toBe(false);
    expect(h.reranker.running).toBe(true);
  });

  test("unload reranker stops only the reranker child", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/unload", { model: "reranker" });
    expect(response.status).toBe(200);
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(false);
  });

  test("an empty unload body stops both children", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/unload", {});
    expect(response.status).toBe(200);
    expect(h.embedding.running).toBe(false);
    expect(h.reranker.running).toBe(false);
  });

  test('unload "all" stops both children', async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/unload", { model: "all" });
    expect(response.status).toBe(200);
    expect(h.embedding.running).toBe(false);
    expect(h.reranker.running).toBe(false);
  });

  test("rejects an unknown unload target and leaves both children up", async () => {
    const h = makeService();
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const response = await admin(h, "/admin/unload", { model: "everything" });
    expect(response.status).toBe(400);
    const body = (await response.json()) as ErrorBody;
    expect(body.error.type).toBe("invalid_request_error");
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);
  });

  test("reports both children independently on /health", async () => {
    const h = makeService({ embeddingTtlSeconds: 60, rerankerTtlSeconds: 45 });
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });

    const body = await health(h);
    expect(body.ok).toBe(true);
    expect(body.mode).toBe("background");
    expect(body.dimensions).toBe(2);
    expect(body.modelId).toBe(h.cfg.modelId);
    expect(body.upstream).toMatchObject({ state: "running", kind: "embedding" });
    expect(body.lastRequestAt).not.toBeNull();
    expect(body.idleTtlMs).toBe(60_000);
    expect(body.reranker.enabled).toBe(true);
    expect(body.reranker.upstream).toMatchObject({ state: "running", kind: "reranker" });
    expect(body.reranker.lastRequestAt).not.toBeNull();
    expect(body.reranker.idleTtlMs).toBe(45_000);
    expect(body.reranker.modelId).toBe(h.cfg.reranker.modelId);

    await admin(h, "/admin/unload", { model: "embedding" });
    const stopped = await health(h);
    expect(stopped.upstream.state).toBe("stopped");
    expect(stopped.reranker.upstream.state).toBe("running");
    expect(stopped.reranker.lastRequestAt).not.toBeNull();
  });
});

describe("EmbedService in-flight lifecycle", () => {
  test("does not reap a child while a cold request is in flight even when the ttl is shorter than the start", async () => {
    const h = makeService({
      rerankerTtlSeconds: 0.05,
      rerankerStartDelayMs: 300,
      rerankResponseDelayMs: 150,
    });
    const response = await rerank(h, { query: "q", documents: ["a"] });
    expect(response.status).toBe(200);
    expect(h.reranker.starts).toBe(1);
    // The cold start plus the forwarded request outlived the ttl; the child
    // serving that request must not be unloaded mid-flight.
    expect(h.reranker.stops).toBe(0);
    expect(h.reranker.running).toBe(true);

    // Once the request has completed, the normal idle window unloads it.
    await Bun.sleep(250);
    expect(h.reranker.stops).toBe(1);
    expect(h.reranker.running).toBe(false);
  });

  test("a mode switch during an embedding request stops the child after the request completes", async () => {
    const h = makeService({
      embeddingStartDelayMs: 300,
      embeddingResponseDelayMs: 150,
    });
    await embed(h, { input: ["a"], dimensions: 2 });
    await rerank(h, { query: "q", documents: ["a"] });
    expect(h.embedding.running).toBe(true);
    expect(h.reranker.running).toBe(true);

    const inFlight = embed(h, { input: ["a"], dimensions: 2 });
    // The fake has the forwarded request, so the service has it in flight.
    await waitFor(() => h.embeddingHttp.requests.length === 2);

    const modeResponse = await admin(h, "/admin/mode", { mode: "throughput" });
    expect(modeResponse.status).toBe(200);
    // The in-flight request is not yanked; the stop is pending.
    expect(h.embedding.stops).toBe(0);
    expect(h.embedding.running).toBe(true);
    expect(h.service.currentMode).toBe("throughput");

    const response = await inFlight;
    expect(response.status).toBe(200);
    // The pending stop is applied as soon as the request finishes.
    expect(h.embedding.stops).toBe(1);
    expect(h.embedding.running).toBe(false);
    // The reranker is untouched by the embedding mode switch.
    expect(h.reranker.running).toBe(true);
    expect(h.reranker.stops).toBe(0);
    expect(h.reranker.starts).toBe(1);
  });

  test("a mode switch during a cold start stops the child so the next request uses the new mode", async () => {
    const h = makeService({
      embeddingStartDelayMs: 300,
      embeddingResponseDelayMs: 100,
    });
    await rerank(h, { query: "q", documents: ["a"] });
    expect(h.reranker.running).toBe(true);

    const cold = embed(h, { input: ["a"], dimensions: 2 });
    // The child is still health-waiting, so the mode switch cannot stop it yet.
    await waitFor(() => h.embedding.status().state === "starting");

    const modeResponse = await admin(h, "/admin/mode", { mode: "throughput" });
    expect(modeResponse.status).toBe(200);
    expect(h.service.currentMode).toBe("throughput");

    const response = await cold;
    expect(response.status).toBe(200);
    // It cold-started with the old profile, so the pending stop is applied once
    // the request finishes; the next request relaunches it with the new mode.
    expect(h.embedding.stops).toBe(1);
    expect(h.embedding.running).toBe(false);
    expect(h.embedding.mode).toBe("background");
    expect(h.reranker.running).toBe(true);
    expect(h.reranker.stops).toBe(0);
    expect(h.reranker.starts).toBe(1);

    await embed(h, { input: ["a"], dimensions: 2 });
    expect(h.embedding.starts).toBe(2);
    expect(h.embedding.mode).toBe("throughput");
  });
});
