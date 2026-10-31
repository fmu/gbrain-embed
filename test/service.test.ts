import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig, type EmbedConfig, type Mode } from "../src/config.ts";
import { EmbedService } from "../src/service.ts";
import { decodeEmbeddingBase64 } from "../src/truncate.ts";
import type { UpstreamLike, UpstreamStatus } from "../src/upstream.ts";

class FakeUpstream implements UpstreamLike {
  starts = 0;
  running = false;
  mode: Mode = "background";
  startedAt: number | null = null;

  async start(mode: Mode): Promise<void> {
    this.starts += 1;
    this.running = true;
    this.mode = mode;
    this.startedAt = Date.now();
  }

  async stop(): Promise<void> {
    this.running = false;
    this.startedAt = null;
  }

  isRunning(): boolean {
    return this.running;
  }

  status(): UpstreamStatus {
    return {
      state: this.running ? "running" : "stopped",
      ...(this.running && this.startedAt !== null ? { startedAt: this.startedAt } : {}),
      mode: this.mode,
    };
  }
}

function startFakeUpstream(vectors: number[][], status = 200) {
  return Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname !== "/v1/embeddings") {
        return new Response("not found", { status: 404 });
      }
      if (status !== 200) {
        return new Response(JSON.stringify({ error: "boom" }), {
          status,
          headers: { "content-type": "application/json" },
        });
      }
      const body = (await request.json()) as { input?: unknown };
      const count = Array.isArray(body.input) ? body.input.length : 1;
      const data = Array.from({ length: count }, (_, index) => ({
        object: "embedding",
        embedding: vectors[index % vectors.length],
        index,
      }));
      return new Response(
        JSON.stringify({ object: "list", data, model: "fake", usage: { prompt_tokens: count } }),
        { headers: { "content-type": "application/json" } },
      );
    },
  });
}

let temp: string;
let fake: ReturnType<typeof startFakeUpstream>;
let cfg: EmbedConfig;
let upstream: FakeUpstream;
let service: EmbedService;
let base: string;

function makeService(vectors: number[][], status = 200): void {
  temp = mkdtempSync(join(tmpdir(), "gbrain-embed-test-"));
  fake = startFakeUpstream(vectors, status);
  cfg = defaultConfig();
  cfg.listenPort = 0;
  cfg.upstreamPort = fake.port;
  cfg.stateDir = temp;
  cfg.modelPath = join(temp, "model.gguf");
  cfg.nativeDimensions = 4;
  cfg.dimensions = 2;
  cfg.idleCheckMs = 20;
  cfg.modes.background.ttlSeconds = 0.2;
  cfg.modes.throughput.ttlSeconds = 0.2;
  upstream = new FakeUpstream();
  service = new EmbedService(cfg, upstream);
  service.start();
  base = `http://127.0.0.1:${service.port}`;
}

async function embed(body: unknown): Promise<Response> {
  return fetch(`${base}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

afterEach(async () => {
  await service.stop();
  fake.stop(true);
  rmSync(temp, { recursive: true, force: true });
});

describe("EmbedService", () => {
  test("starts the model server once and fits the vector to the configured width", async () => {
    makeService([[3, 4, 0, 0]]);
    const response = await embed({ input: ["a"], dimensions: 2 });
    expect(response.status).toBe(200);
    const body = (await response.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([0.6, 0.8]);
    expect(upstream.starts).toBe(1);
  });

  test("concurrent cold requests start the child exactly once", async () => {
    makeService([[3, 4, 0, 0]]);
    const responses = await Promise.all(
      Array.from({ length: 5 }, () => embed({ input: ["a"], dimensions: 2 })),
    );
    for (const response of responses) expect(response.status).toBe(200);
    expect(upstream.starts).toBe(1);
  });

  test("a model listing does not wake the model", async () => {
    makeService([[3, 4, 0, 0]]);
    const response = await fetch(`${base}/v1/models`);
    expect(response.status).toBe(200);
    expect(upstream.starts).toBe(0);
  });

  test("honours a request for the native width without truncating", async () => {
    makeService([[3, 4, 0, 0]]);
    const response = await embed({ input: ["a"], dimensions: 4 });
    const body = (await response.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([3, 4, 0, 0]);
  });

  test("switching mode stops the child so the next request uses the new profile", async () => {
    makeService([[3, 4, 0, 0]]);
    await embed({ input: ["a"], dimensions: 2 });
    expect(upstream.running).toBe(true);
    const response = await fetch(`${base}/admin/mode`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode: "throughput" }),
    });
    expect(response.status).toBe(200);
    expect(upstream.running).toBe(false);
    await embed({ input: ["a"], dimensions: 2 });
    expect(upstream.mode).toBe("throughput");
  });

  test("stops the child after the idle ttl", async () => {
    makeService([[3, 4, 0, 0]]);
    await embed({ input: ["a"], dimensions: 2 });
    expect(upstream.running).toBe(true);
    await Bun.sleep(400);
    expect(upstream.running).toBe(false);
  });

  test("surfaces an upstream error without pretending it succeeded", async () => {
    makeService([[3, 4, 0, 0]], 500);
    const response = await embed({ input: ["a"], dimensions: 2 });
    expect(response.status).toBe(500);
    expect(await response.text()).toContain("boom");
  });

  test("encodes base64 when the caller asks for it", async () => {
    makeService([[3, 4, 0, 0]]);
    const response = await embed({ input: ["a"], dimensions: 2, encoding_format: "base64" });
    const body = (await response.json()) as { data: Array<{ embedding: string }> };
    expect(typeof body.data[0].embedding).toBe("string");
    const decoded = decodeEmbeddingBase64(body.data[0].embedding);
    expect(decoded[0]).toBeCloseTo(0.6, 6);
    expect(decoded[1]).toBeCloseTo(0.8, 6);
  });

  test("reports mode and child state on /health", async () => {
    makeService([[3, 4, 0, 0]]);
    const response = await fetch(`${base}/health`);
    const body = (await response.json()) as {
      ok: boolean;
      mode: string;
      dimensions: number;
      upstream: { state: string };
    };
    expect(body.ok).toBe(true);
    expect(body.mode).toBe("background");
    expect(body.dimensions).toBe(2);
    expect(body.upstream.state).toBe("stopped");
  });
});
