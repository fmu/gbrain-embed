#!/usr/bin/env bun
/**
 * `gbrain-embed` CLI.
 *
 * The launchd agent runs `serve`; everything else is operator-facing. The CLI
 * is deliberately dependency-free beyond Bun and the Homebrew llama.cpp build.
 */
import { loadConfig, listenBase, type EmbedConfig, type Mode } from "./config.ts";
import { installModel, modelDescriptors, type ModelKey } from "./model.ts";
import { EmbedService, readMode, writeMode, type UnloadTarget } from "./service.ts";
import { Upstream, type UpstreamStatus } from "./upstream.ts";

const QUERY_PREFIX =
  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

interface RerankStatusView {
  enabled: boolean;
  upstream: UpstreamStatus;
  lastRequestAt: number | null;
  idleTtlMs: number;
  modelPath: string;
  modelId: string;
  upstreamPort: number;
  rssBytes: number | null;
}

interface Health {
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
  reranker?: RerankStatusView;
}

const MODEL_KEYS: ModelKey[] = ["embedding", "reranker"];

function usage(): string {
  const cfg = loadConfig();
  return `gbrain-embed — on-demand Qwen3 embeddings and reranking for gbrain

Usage:
  gbrain-embed serve                     Run the loopback proxy (launchd runs this)
  gbrain-embed status [--json]           Show mode, child states, last requests, RSS
  gbrain-embed mode <background|throughput> [--json]
                                         Switch the embedding resource profile
  gbrain-embed model install [embedding|reranker|all] [--refresh] [--source <ollama:tag|hf:repo/file|path>] [--json]
                                         Hardlink/download the verified GGUFs
  gbrain-embed setup [--json]            Install the models, configure gbrain, backfill
  gbrain-embed bench [--model <embedding|reranker|all>] [--mode <m>] [--texts N] [--json]
                                         Measure cold start, warm throughput and RSS
  gbrain-embed restart [--model <embedding|reranker|all>] [--json]
                                         Restart the launchd agent (or drop a child)

Global: --config <path>  (or GBRAIN_EMBED_CONFIG)

Models:
  embedding  ${cfg.modelId} at ${cfg.modelPath}
             ttl ${cfg.modes[cfg.defaultMode].ttlSeconds}s in ${cfg.defaultMode} mode; upstream 127.0.0.1:${cfg.upstreamPort}
  reranker   ${cfg.reranker.enabled ? cfg.reranker.modelId : "(disabled)"} at ${cfg.reranker.modelPath}
             ttl ${cfg.reranker.ttlSeconds}s (independent of the embedding mode); upstream 127.0.0.1:${cfg.reranker.upstreamPort}

Profiles (embedding only):
  background   ttl ${cfg.modes.background.ttlSeconds}s, ${cfg.modes.background.llamaArgs.join(" ")}, taskpolicy -b
  throughput   ttl ${cfg.modes.throughput.ttlSeconds}s, ${cfg.modes.throughput.llamaArgs.join(" ")}, normal QoS
`;
}

function parseArgs(argv: string[]): { command: string; rest: string[]; config?: string } {
  let config: string | undefined;
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--config") {
      config = argv[++i];
      continue;
    }
    rest.push(arg);
  }
  return { command: rest[0] ?? "", rest: rest.slice(1), config };
}

function flag(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index === -1 ? undefined : args[index + 1];
}

function has(args: string[], name: string): boolean {
  return args.includes(name);
}

function printJson(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function humanBytes(bytes: number | null): string {
  if (bytes === null) return "n/a";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

async function fetchHealth(cfg: EmbedConfig, timeoutMs = 2_000): Promise<Health | null> {
  try {
    const response = await fetch(`${listenBase(cfg)}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) return null;
    return (await response.json()) as Health;
  } catch {
    return null;
  }
}

function runGbrain(cfg: EmbedConfig, args: string[]): { code: number; out: string; err: string } {
  const result = Bun.spawnSync({
    cmd: [cfg.gbrainBin, ...args],
    stdout: "pipe",
    stderr: "pipe",
  });
  return {
    code: result.exitCode ?? 1,
    out: result.stdout.toString(),
    err: result.stderr.toString(),
  };
}

function mustGbrain(cfg: EmbedConfig, args: string[], quiet = false): string {
  const result = runGbrain(cfg, args);
  if (!quiet) {
    if (result.out) process.stdout.write(result.out);
    if (result.err) process.stderr.write(result.err);
  }
  if (result.code !== 0) {
    const combined = result.out + result.err;
    const hint = /pglite_busy|already open through|writer lock/i.test(combined)
      ? "\nHint: a running `gbrain serve` holds the PGlite writer lock. Stop it, re-run this step, then start it again."
      : "";
    throw new Error(`gbrain ${args.join(" ")} exited with ${result.code}${hint}`);
  }
  return result.out;
}

function captureGbrain(cfg: EmbedConfig, args: string[]): { code: number; out: string; err: string } {
  return runGbrain(cfg, args);
}

async function commandServe(cfg: EmbedConfig): Promise<void> {
  const service = new EmbedService(cfg, new Upstream(cfg, "embedding"), new Upstream(cfg, "reranker"));
  try {
    service.start();
  } catch (error) {
    throw new Error(
      `could not listen on ${listenBase(cfg)}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  console.log(
    `[gbrain-embed] listening on ${listenBase(cfg)} (mode ${service.currentMode}, ` +
      `embedding ${cfg.upstreamHost}:${cfg.upstreamPort}, ` +
      `reranker ${cfg.reranker.enabled ? `${cfg.upstreamHost}:${cfg.reranker.upstreamPort}` : "disabled"}, ` +
      `${cfg.dimensions}d)`,
  );
  const shutdown = async (): Promise<void> => {
    console.log("[gbrain-embed] shutting down");
    await service.stop();
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown());
  process.on("SIGINT", () => void shutdown());
  await new Promise(() => {});
}

async function commandStatus(cfg: EmbedConfig, args: string[]): Promise<void> {
  const health = await fetchHealth(cfg);
  const json = has(args, "--json");
  if (!health) {
    const mode = readMode(cfg);
    const body = {
      ok: false,
      mode,
      upstream: { state: "stopped", kind: "embedding" },
      lastRequestAt: null,
      idleTtlMs: cfg.modes[mode].ttlSeconds * 1000,
      modelPath: cfg.modelPath,
      modelId: cfg.modelId,
      dimensions: cfg.dimensions,
      nativeDimensions: cfg.nativeDimensions,
      listenPort: cfg.listenPort,
      upstreamPort: cfg.upstreamPort,
      rssBytes: null,
      reranker: {
        enabled: cfg.reranker.enabled,
        upstream: { state: "stopped", kind: "reranker" },
        lastRequestAt: null,
        idleTtlMs: cfg.reranker.ttlSeconds * 1000,
        modelPath: cfg.reranker.modelPath,
        modelId: cfg.reranker.modelId,
        upstreamPort: cfg.reranker.upstreamPort,
        rssBytes: null,
      },
    };
    if (json) printJson(body);
    else {
      console.log(`gbrain-embed: not running (${listenBase(cfg)})`);
      console.log(`mode: ${body.mode}; embedding: ${cfg.modelId} (${cfg.dimensions}d)`);
      console.log(`reranker: ${body.reranker.enabled ? cfg.reranker.modelId : "disabled"}`);
    }
    return;
  }
  if (json) {
    printJson(health);
    return;
  }
  console.log(`gbrain-embed: running on ${listenBase(cfg)} (mode ${health.mode})`);
  console.log(
    `embedding: ${health.upstream.state}${health.upstream.pid ? ` pid ${health.upstream.pid}` : ""}; ` +
      `ttl ${Math.round(health.idleTtlMs / 1000)}s; rss ${humanBytes(health.rssBytes)}`,
  );
  console.log(`  model: ${health.modelId} (${health.dimensions}d of ${health.nativeDimensions}) at ${health.modelPath}`);
  console.log(
    `  last request: ${health.lastRequestAt ? new Date(health.lastRequestAt).toISOString() : "never"}`,
  );
  const reranker = health.reranker;
  if (reranker) {
    console.log(
      `reranker: ${reranker.enabled ? reranker.upstream.state : "disabled"}` +
        `${reranker.upstream.pid ? ` pid ${reranker.upstream.pid}` : ""}; ` +
        `ttl ${Math.round(reranker.idleTtlMs / 1000)}s; rss ${humanBytes(reranker.rssBytes)}`,
    );
    console.log(`  model: ${reranker.modelId} at ${reranker.modelPath}`);
    console.log(
      `  last request: ${reranker.lastRequestAt ? new Date(reranker.lastRequestAt).toISOString() : "never"}`,
    );
  }
}

async function commandMode(cfg: EmbedConfig, args: string[]): Promise<void> {
  const mode = args.find((arg) => arg === "background" || arg === "throughput");
  if (!mode) throw new Error("usage: gbrain-embed mode <background|throughput>");
  writeMode(cfg, mode);
  const json = has(args, "--json");
  try {
    const response = await fetch(`${listenBase(cfg)}/admin/mode`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode }),
      signal: AbortSignal.timeout(5_000),
    });
    if (!response.ok) throw new Error(await response.text());
    const health = (await response.json()) as Health;
    if (json) printJson(health);
    else {
      console.log(
        `mode: ${mode} (embedding child ${health.upstream.state}; next request starts the new profile; ` +
          "the reranker is unaffected)",
      );
    }
  } catch (error) {
    if (json) printJson({ mode, applied: false, reason: error instanceof Error ? error.message : String(error) });
    else console.log(`mode written: ${mode}; the proxy is not running, so it applies on next start`);
  }
}

async function commandModelInstall(cfg: EmbedConfig, args: string[]): Promise<void> {
  const requested = args.find((arg) => arg === "embedding" || arg === "reranker" || arg === "all");
  const target = requested ?? "all";
  const source = flag(args, "--source");
  if (source && target === "all") {
    throw new Error(
      "pick one model for --source: gbrain-embed model install <embedding|reranker> --source ...",
    );
  }
  const keys: ModelKey[] = target === "all" ? MODEL_KEYS : [target as ModelKey];
  const results: Partial<Record<ModelKey, unknown>> = {};
  for (const key of keys) {
    const result = await installModel(cfg, key, {
      refresh: has(args, "--refresh"),
      source,
    });
    results[key] = result;
    if (!has(args, "--json")) {
      console.log(`${modelDescriptors(cfg)[key].label}: ${result.path}`);
      console.log(`  sha256: ${result.sha256}`);
      console.log(`  source: ${result.source}; ${(result.bytes / 1024 ** 3).toFixed(2)} GiB`);
    }
  }
  if (has(args, "--json")) printJson(target === "all" ? results : results[keys[0]]);
}

async function commandSetup(cfg: EmbedConfig, args: string[]): Promise<void> {
  const json = has(args, "--json");
  const models = await installModel(cfg, "embedding", {});
  const rerankerModel = cfg.reranker.enabled
    ? await installModel(cfg, "reranker", {})
    : null;
  if (!json) {
    console.log(`embedding model ready: ${models.path}`);
    console.log(`  sha256: ${models.sha256}`);
    if (rerankerModel) {
      console.log(`reranker model ready: ${rerankerModel.path}`);
      console.log(`  sha256: ${rerankerModel.sha256}`);
    }
  }

  let temporary: EmbedService | null = null;
  let health = await fetchHealth(cfg);
  if (!health) {
    temporary = new EmbedService(cfg, new Upstream(cfg, "embedding"), new Upstream(cfg, "reranker"));
    temporary.start();
    health = await fetchHealth(cfg);
    if (!health) throw new Error("started the embedding proxy but /health did not answer");
    if (!json) console.log(`proxy started for setup on ${listenBase(cfg)}`);
  }

  try {
    mustGbrain(cfg, [
      "config", "set", "provider_base_urls.llama-server", `${listenBase(cfg)}/v1`,
    ], json);

    const dims = captureGbrain(cfg, ["config", "get", "embedding_dimensions"]);
    const current = dims.out.trim();
    if (dims.code === 0 && current && current !== String(cfg.dimensions)) {
      throw new Error(
        `the brain's embedding_dimensions is ${current}, but this service emits ${cfg.dimensions}. ` +
          "Changing the width is a re-embed: preview it with `gbrain migrate embeddings --to " +
          `llama-server:${cfg.modelId} --dim ${cfg.dimensions} --dry-run` +
          " and get approval before applying.",
      );
    }

    const enabled = captureGbrain(cfg, [
      "embeddings", "enable",
      "--embedding-model", `llama-server:${cfg.modelId}`,
      "--embedding-dimensions", String(cfg.dimensions),
      "--json",
    ]);
    if (enabled.code !== 0) {
      if (!json) {
        if (enabled.out) process.stdout.write(enabled.out);
        if (enabled.err) process.stderr.write(enabled.err);
      }
      const combined = enabled.out + enabled.err;
      const hint = /pglite_busy|already open through|writer lock/i.test(combined)
        ? "\nHint: stop the running `gbrain serve`, re-run `gbrain-embed setup`, then start it again."
        : "";
      throw new Error(`gbrain embeddings enable exited with ${enabled.code}${hint}`);
    }

    mustGbrain(cfg, ["config", "set", "embedding_query_prefix", QUERY_PREFIX], json);

    if (cfg.reranker.enabled) {
      mustGbrain(cfg, [
        "config", "set", "provider_base_urls.llama-server-reranker", `${listenBase(cfg)}/v1`,
      ], json);
      mustGbrain(cfg, [
        "config", "set", "search.reranker.model", `llama-server-reranker:${cfg.reranker.modelId}`,
      ], json);
      mustGbrain(cfg, ["config", "set", "search.reranker.enabled", "true"], json);
    } else {
      mustGbrain(cfg, ["config", "set", "search.reranker.enabled", "false"], json);
    }

    mustGbrain(cfg, ["embed", "--stale", "--catch-up"], json);
    mustGbrain(cfg, ["doctor", "--only", "embeddings", "--json"], json);

    const summary = {
      status: "enabled",
      model: `llama-server:${cfg.modelId}`,
      dimensions: cfg.dimensions,
      modelPath: models.path,
      sha256: models.sha256,
      listen: `${listenBase(cfg)}/v1`,
      mode: readMode(cfg),
      reranker: cfg.reranker.enabled
        ? {
            enabled: true,
            model: `llama-server-reranker:${cfg.reranker.modelId}`,
            modelPath: rerankerModel?.path ?? cfg.reranker.modelPath,
            sha256: rerankerModel?.sha256 ?? null,
            listen: `${listenBase(cfg)}/v1`,
          }
        : { enabled: false },
      enable: enabled.out.trim() ? safeJson(enabled.out) : null,
    };
    if (json) printJson(summary);
    else {
      console.log(`embeddings enabled: llama-server:${cfg.modelId} (${cfg.dimensions}d)`);
      if (cfg.reranker.enabled) {
        console.log(`reranker enabled: llama-server-reranker:${cfg.reranker.modelId}`);
      }
      console.log(`endpoint: ${listenBase(cfg)}/v1; mode: ${summary.mode}`);
    }
  } finally {
    if (temporary) await temporary.stop();
  }
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function embedBatch(cfg: EmbedConfig, texts: string[]): Promise<number[][]> {
  const response = await fetch(`${listenBase(cfg)}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: cfg.modelId, input: texts, dimensions: cfg.dimensions }),
    signal: AbortSignal.timeout(cfg.requestTimeoutMs),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`embedding request failed (${response.status}): ${text}`);
  const body = JSON.parse(text) as { data?: Array<{ embedding?: number[] }> };
  if (!Array.isArray(body.data) || body.data.length !== texts.length) {
    throw new Error(`embedding response held ${body.data?.length ?? 0} vectors for ${texts.length} inputs`);
  }
  return body.data.map((item) => item.embedding ?? []);
}

interface RerankItem {
  index: number;
  relevance_score: number;
}

async function rerankBatch(
  cfg: EmbedConfig,
  query: string,
  documents: string[],
): Promise<RerankItem[]> {
  const response = await fetch(`${listenBase(cfg)}/v1/rerank`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: cfg.reranker.modelId,
      query,
      documents,
    }),
    signal: AbortSignal.timeout(cfg.requestTimeoutMs),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`rerank request failed (${response.status}): ${text}`);
  const body = JSON.parse(text) as { results?: RerankItem[] };
  if (!Array.isArray(body.results) || body.results.length !== documents.length) {
    throw new Error(
      `rerank response held ${body.results?.length ?? 0} scores for ${documents.length} documents`,
    );
  }
  return body.results;
}

async function unload(cfg: EmbedConfig, model: UnloadTarget): Promise<void> {
  await fetch(`${listenBase(cfg)}/admin/unload`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model }),
    signal: AbortSignal.timeout(10_000),
  });
}

function benchTexts(count: number): string[] {
  return Array.from({ length: count }, (_, index) =>
    (`The appellant argues that the lower court erred in its assessment of the evidence. `.repeat(8) +
      ` [doc ${index}]`).slice(0, 1000),
  );
}

async function benchEmbeddings(cfg: EmbedConfig, texts: string[], count: number): Promise<unknown> {
  await unload(cfg, "embedding");
  const coldStart = Date.now();
  await embedBatch(cfg, texts.slice(0, 2));
  const coldStartMs = Date.now() - coldStart;

  const warm: number[] = [];
  for (let i = 0; i < 3; i += 1) {
    const started = Date.now();
    const vectors = await embedBatch(cfg, texts);
    const elapsed = Date.now() - started;
    if (vectors.some((vector) => vector.length !== cfg.dimensions)) {
      throw new Error(`a vector did not have ${cfg.dimensions} dimensions`);
    }
    warm.push(elapsed);
  }
  warm.sort((a, b) => a - b);
  const median = warm[1];
  const after = await fetchHealth(cfg);
  return {
    texts: count,
    coldStartMs,
    warmMs: median,
    textsPerSecond: count / (median / 1000),
    rssBytes: after?.rssBytes ?? null,
    upstream: after?.upstream ?? null,
  };
}

async function benchReranker(cfg: EmbedConfig, documents: string[], count: number): Promise<unknown> {
  const query = "Does the lower court's assessment of the evidence withstand appeal?";
  await unload(cfg, "reranker");
  const coldStart = Date.now();
  await rerankBatch(cfg, query, documents.slice(0, 10));
  const coldStartMs = Date.now() - coldStart;

  const warm: number[] = [];
  for (let i = 0; i < 3; i += 1) {
    const started = Date.now();
    const results = await rerankBatch(cfg, query, documents);
    const elapsed = Date.now() - started;
    if (results.some((result) => typeof result.relevance_score !== "number")) {
      throw new Error("a rerank result did not carry a numeric relevance_score");
    }
    warm.push(elapsed);
  }
  warm.sort((a, b) => a - b);
  const median = warm[1];
  const after = await fetchHealth(cfg);
  return {
    documents: count,
    coldStartMs,
    warmMs: median,
    documentsPerSecond: count / (median / 1000),
    rssBytes: after?.reranker?.rssBytes ?? null,
    upstream: after?.reranker?.upstream ?? null,
  };
}

async function commandBench(cfg: EmbedConfig, args: string[]): Promise<void> {
  const mode = flag(args, "--mode");
  if (mode === "background" || mode === "throughput") {
    writeMode(cfg, mode);
    try {
      await fetch(`${listenBase(cfg)}/admin/mode`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode }),
        signal: AbortSignal.timeout(5_000),
      });
    } catch {
      /* the proxy may not be running; the bench will start its own */
    }
  }
  const requested = flag(args, "--model") ?? "all";
  if (requested !== "all" && requested !== "embedding" && requested !== "reranker") {
    throw new Error("--model must be embedding, reranker or all");
  }
  if (requested !== "embedding" && !cfg.reranker.enabled) {
    throw new Error("the reranker is disabled in the config");
  }
  const health = await fetchHealth(cfg);
  if (!health) {
    throw new Error(`the proxy is not running; start it with \`gbrain-embed serve\` or activate the launchd agent`);
  }

  const count = Number(flag(args, "--texts") ?? 32);
  const texts = benchTexts(Number.isFinite(count) && count > 0 ? count : 32);
  const result: Record<string, unknown> = { mode: health.mode };
  if (requested === "all" || requested === "embedding") {
    result.embeddings = await benchEmbeddings(cfg, texts, texts.length);
  }
  if (requested === "all" || requested === "reranker") {
    result.reranking = await benchReranker(cfg, texts, texts.length);
  }

  if (has(args, "--json")) {
    printJson(result);
    return;
  }
  console.log(`mode: ${String(result.mode)}`);
  const embeddings = result.embeddings as
    | { texts: number; coldStartMs: number; warmMs: number; textsPerSecond: number; rssBytes: number | null }
    | undefined;
  if (embeddings) {
    console.log(`embedding: ${embeddings.texts} texts`);
    console.log(`  cold start: ${(embeddings.coldStartMs / 1000).toFixed(1)}s`);
    console.log(
      `  warm: ${(embeddings.warmMs / 1000).toFixed(2)}s (${embeddings.textsPerSecond.toFixed(2)} texts/s)`,
    );
    console.log(`  rss: ${humanBytes(embeddings.rssBytes)}`);
  }
  const reranking = result.reranking as
    | { documents: number; coldStartMs: number; warmMs: number; documentsPerSecond: number; rssBytes: number | null }
    | undefined;
  if (reranking) {
    console.log(`reranker: ${reranking.documents} documents`);
    console.log(`  cold start: ${(reranking.coldStartMs / 1000).toFixed(1)}s`);
    console.log(
      `  warm: ${(reranking.warmMs / 1000).toFixed(2)}s (${reranking.documentsPerSecond.toFixed(2)} docs/s)`,
    );
    console.log(`  rss: ${humanBytes(reranking.rssBytes)}`);
  }
}

async function commandRestart(cfg: EmbedConfig, args: string[]): Promise<void> {
  const requested = flag(args, "--model") ?? "all";
  if (requested !== "all" && requested !== "embedding" && requested !== "reranker") {
    throw new Error("--model must be embedding, reranker or all");
  }
  const uid = typeof process.getuid === "function" ? process.getuid() : 0;
  const result = Bun.spawnSync({
    cmd: ["launchctl", "kickstart", "-k", `gui/${uid}/name.fmu.gbrain-embed`],
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode === 0) {
    if (has(args, "--json")) printJson({ restarted: "launchd", label: "name.fmu.gbrain-embed", model: requested });
    else console.log(`restarted name.fmu.gbrain-embed (${requested})`);
    return;
  }
  try {
    if (requested === "all") {
      await fetch(`${listenBase(cfg)}/admin/restart`, {
        method: "POST",
        signal: AbortSignal.timeout(10_000),
      });
    } else {
      await unload(cfg, requested);
    }
    if (has(args, "--json")) printJson({ restarted: "proxy", label: "name.fmu.gbrain-embed", model: requested });
    else console.log(`asked the running proxy to drop its ${requested} child`);
  } catch (error) {
    throw new Error(
      `launchctl kickstart failed (${result.stderr.toString().trim()}) and the proxy is unreachable: ` +
        (error instanceof Error ? error.message : String(error)),
    );
  }
}

async function main(argv: string[]): Promise<void> {
  const parsed = parseArgs(argv);
  if (parsed.config) process.env.GBRAIN_EMBED_CONFIG = parsed.config;
  const cfg = loadConfig();
  const { command, rest } = parsed;

  if (!command || command === "--help" || command === "-h" || command === "help") {
    process.stdout.write(usage());
    return;
  }
  switch (command) {
    case "serve":
      await commandServe(cfg);
      return;
    case "status":
      await commandStatus(cfg, rest);
      return;
    case "mode":
      await commandMode(cfg, rest);
      return;
    case "model": {
      const sub = rest[0];
      if (sub !== "install") throw new Error("usage: gbrain-embed model install [embedding|reranker|all] [--refresh] [--source ...]");
      await commandModelInstall(cfg, rest.slice(1));
      return;
    }
    case "setup":
      await commandSetup(cfg, rest);
      return;
    case "bench":
      await commandBench(cfg, rest);
      return;
    case "restart":
      await commandRestart(cfg, rest);
      return;
    default:
      throw new Error(`unknown command ${command}; run gbrain-embed --help`);
  }
}

if (import.meta.main) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(`gbrain-embed: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}

export { main };
