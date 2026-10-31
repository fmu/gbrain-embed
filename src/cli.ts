#!/usr/bin/env bun
/**
 * `gbrain-embed` CLI.
 *
 * The launchd agent runs `serve`; everything else is operator-facing. The CLI
 * is deliberately dependency-free beyond Bun and the Homebrew llama.cpp build.
 */
import { loadConfig, listenBase, type EmbedConfig, type Mode } from "./config.ts";
import { installModel, MODEL_SHA256 } from "./model.ts";
import { EmbedService, readMode, writeMode } from "./service.ts";
import { Upstream } from "./upstream.ts";

const QUERY_PREFIX =
  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

interface Health {
  ok: true;
  mode: Mode;
  upstream: { state: string; pid?: number; startedAt?: number; mode?: Mode };
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

function usage(): string {
  return `gbrain-embed — on-demand Qwen3 embeddings for gbrain

Usage:
  gbrain-embed serve                     Run the loopback proxy (launchd runs this)
  gbrain-embed status [--json]           Show mode, child state, last request, RSS
  gbrain-embed mode <background|throughput> [--json]
                                         Switch the resource profile
  gbrain-embed model install [--refresh] [--source <ollama:tag|hf:repo/file|path>] [--json]
                                         Hardlink/download the verified Q6_K GGUF
  gbrain-embed setup [--json]            Install the model, configure gbrain, backfill
  gbrain-embed bench [--mode <m>] [--texts N] [--json]
                                         Measure cold start and warm texts/s
  gbrain-embed restart [--json]          Restart the launchd agent (or the proxy)

Global: --config <path>  (or GBRAIN_EMBED_CONFIG)

Profiles:
  background   ttl ${loadConfig().modes.background.ttlSeconds}s, ${loadConfig().modes.background.llamaArgs.join(" ")}, taskpolicy -b
  throughput   ttl ${loadConfig().modes.throughput.ttlSeconds}s, ${loadConfig().modes.throughput.llamaArgs.join(" ")}, normal QoS
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
  const service = new EmbedService(cfg, new Upstream(cfg));
  try {
    service.start();
  } catch (error) {
    throw new Error(
      `could not listen on ${listenBase(cfg)}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  console.log(
    `[gbrain-embed] listening on ${listenBase(cfg)} (mode ${service.currentMode}, ` +
      `upstream ${cfg.upstreamHost}:${cfg.upstreamPort}, ${cfg.dimensions}d)`,
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
    const body = {
      ok: false,
      mode: readMode(cfg),
      upstream: { state: "stopped" },
      modelPath: cfg.modelPath,
      modelId: cfg.modelId,
      dimensions: cfg.dimensions,
      listenPort: cfg.listenPort,
      upstreamPort: cfg.upstreamPort,
    };
    if (json) printJson(body);
    else {
      console.log(`gbrain-embed: not running (${listenBase(cfg)})`);
      console.log(`mode: ${body.mode}; model: ${cfg.modelId} (${cfg.dimensions}d)`);
    }
    return;
  }
  if (json) {
    printJson(health);
    return;
  }
  console.log(`gbrain-embed: running on ${listenBase(cfg)} (mode ${health.mode})`);
  console.log(`upstream: ${health.upstream.state}${health.upstream.pid ? ` pid ${health.upstream.pid}` : ""}`);
  console.log(`model: ${health.modelId} (${health.dimensions}d of ${health.nativeDimensions}) at ${health.modelPath}`);
  console.log(`idle ttl: ${Math.round(health.idleTtlMs / 1000)}s; rss: ${humanBytes(health.rssBytes)}`);
  console.log(`last request: ${health.lastRequestAt ? new Date(health.lastRequestAt).toISOString() : "never"}`);
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
    else console.log(`mode: ${mode} (child ${health.upstream.state}; next request starts the new profile)`);
  } catch (error) {
    if (json) printJson({ mode, applied: false, reason: error instanceof Error ? error.message : String(error) });
    else console.log(`mode written: ${mode}; the proxy is not running, so it applies on next start`);
  }
}

async function commandModelInstall(cfg: EmbedConfig, args: string[]): Promise<void> {
  const result = await installModel(cfg, {
    refresh: has(args, "--refresh"),
    source: flag(args, "--source"),
  });
  if (has(args, "--json")) printJson(result);
  else {
    console.log(`model: ${result.path}`);
    console.log(`sha256: ${result.sha256}`);
    console.log(`source: ${result.source}; ${(result.bytes / 1024 ** 3).toFixed(2)} GiB`);
  }
}

async function commandSetup(cfg: EmbedConfig, args: string[]): Promise<void> {
  const json = has(args, "--json");
  const model = await installModel(cfg, {});
  if (!json) {
    console.log(`model ready: ${model.path}`);
    console.log(`sha256: ${model.sha256}`);
  }

  let temporary: EmbedService | null = null;
  let health = await fetchHealth(cfg);
  if (!health) {
    temporary = new EmbedService(cfg, new Upstream(cfg));
    temporary.start();
    health = await fetchHealth(cfg);
    if (!health) throw new Error("started the embedding proxy but /health did not answer");
    if (!json) console.log(`embedding proxy started for setup on ${listenBase(cfg)}`);
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
    mustGbrain(cfg, ["embed", "--stale", "--catch-up"], json);
    mustGbrain(cfg, ["doctor", "--only", "embeddings", "--json"], json);

    const summary = {
      status: "enabled",
      model: `llama-server:${cfg.modelId}`,
      dimensions: cfg.dimensions,
      modelPath: model.path,
      sha256: model.sha256,
      listen: `${listenBase(cfg)}/v1`,
      mode: readMode(cfg),
      enable: enabled.out.trim() ? safeJson(enabled.out) : null,
    };
    if (json) printJson(summary);
    else {
      console.log(`embeddings enabled: llama-server:${cfg.modelId} (${cfg.dimensions}d)`);
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
  const health = await fetchHealth(cfg);
  if (!health) {
    throw new Error(`the proxy is not running; start it with \`gbrain-embed serve\` or activate the launchd agent`);
  }
  await fetch(`${listenBase(cfg)}/admin/restart`, {
    method: "POST",
    signal: AbortSignal.timeout(10_000),
  });

  const count = Number(flag(args, "--texts") ?? 32);
  const texts = Array.from({ length: count }, (_, index) =>
    (`The appellant argues that the lower court erred in its assessment of the evidence. `.repeat(8) +
      ` [doc ${index}]`).slice(0, 1000),
  );

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
  const result = {
    mode: after?.mode ?? readMode(cfg),
    texts: count,
    coldStartMs,
    warmMs: median,
    textsPerSecond: count / (median / 1000),
    rssBytes: after?.rssBytes ?? null,
    upstream: after?.upstream ?? null,
  };
  if (has(args, "--json")) printJson(result);
  else {
    console.log(`mode: ${result.mode}; texts: ${count}`);
    console.log(`cold start: ${(coldStartMs / 1000).toFixed(1)}s`);
    console.log(`warm: ${(median / 1000).toFixed(2)}s (${result.textsPerSecond.toFixed(2)} texts/s)`);
    console.log(`rss: ${humanBytes(result.rssBytes)}; upstream: ${result.upstream?.state ?? "unknown"}`);
  }
}

async function commandRestart(cfg: EmbedConfig, args: string[]): Promise<void> {
  const uid = typeof process.getuid === "function" ? process.getuid() : 0;
  const result = Bun.spawnSync({
    cmd: ["launchctl", "kickstart", "-k", `gui/${uid}/name.fmu.gbrain-embed`],
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode === 0) {
    if (has(args, "--json")) printJson({ restarted: "launchd", label: "name.fmu.gbrain-embed" });
    else console.log("restarted name.fmu.gbrain-embed");
    return;
  }
  try {
    await fetch(`${listenBase(cfg)}/admin/restart`, {
      method: "POST",
      signal: AbortSignal.timeout(10_000),
    });
    if (has(args, "--json")) printJson({ restarted: "proxy", label: "name.fmu.gbrain-embed" });
    else console.log("asked the running proxy to stop its child");
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
      if (sub !== "install") throw new Error("usage: gbrain-embed model install [--refresh] [--source ...]");
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

export { main, MODEL_SHA256 };
