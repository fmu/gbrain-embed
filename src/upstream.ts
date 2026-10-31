/**
 * A managed `llama-server` child: spawn on demand, wait for `/health`, stop
 * after idle, and never leave an orphan behind.
 *
 * There are two of them, one per model kind: the embedding server
 * (`--embeddings`) and the reranker (`--reranking`). llama.cpp makes those
 * launch flags mutually exclusive, so they are separate processes with
 * separate ports, PID records, logs and idle state. That is also what makes
 * their load/unload cycles independent: stopping one never touches the other.
 */
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { EmbedConfig, Mode } from "./config.ts";
import {
  modelBase,
  rerankerErrLogPath,
  rerankerOutLogPath,
  rerankerStatePath,
  upstreamErrLogPath,
  upstreamOutLogPath,
  upstreamStatePath,
} from "./config.ts";

export type UpstreamKind = "embedding" | "reranker";
export type UpstreamState = "stopped" | "starting" | "running" | "stopping";

export interface UpstreamStatus {
  state: UpstreamState;
  kind: UpstreamKind;
  pid?: number;
  startedAt?: number;
  mode?: Mode;
}

export interface UpstreamLike {
  start(mode: Mode): Promise<void>;
  stop(): Promise<void>;
  isRunning(): boolean;
  status(): UpstreamStatus;
}

interface ChildRecord {
  pid: number;
  startedAt: number;
  mode: Mode;
  kind?: UpstreamKind;
  modelPath: string;
  port: number;
}

const LOG_LIMIT_BYTES = 20 * 1024 * 1024;
const STOP_GRACE_MS = 5_000;
const ORPHAN_GRACE_MS = 2_000;

function pathsFor(cfg: EmbedConfig, kind: UpstreamKind): {
  statePath: string;
  outLogPath: string;
  errLogPath: string;
} {
  return kind === "reranker"
    ? {
        statePath: rerankerStatePath(cfg),
        outLogPath: rerankerOutLogPath(cfg),
        errLogPath: rerankerErrLogPath(cfg),
      }
    : {
        statePath: upstreamStatePath(cfg),
        outLogPath: upstreamOutLogPath(cfg),
        errLogPath: upstreamErrLogPath(cfg),
      };
}

function modelPathFor(cfg: EmbedConfig, kind: UpstreamKind): string {
  return kind === "reranker" ? cfg.reranker.modelPath : cfg.modelPath;
}

function modelIdFor(cfg: EmbedConfig, kind: UpstreamKind): string {
  return kind === "reranker" ? cfg.reranker.modelId : cfg.modelId;
}

function portFor(cfg: EmbedConfig, kind: UpstreamKind): number {
  return kind === "reranker" ? cfg.reranker.upstreamPort : cfg.upstreamPort;
}

/** Full llama.cpp argument vector for one model kind. */
export function buildLlamaArgs(
  cfg: EmbedConfig,
  mode: Mode,
  kind: UpstreamKind = "embedding",
): string[] {
  if (kind === "reranker") {
    return [
      "-m", cfg.reranker.modelPath,
      "--alias", cfg.reranker.modelId,
      "--reranking",
      "--pooling", "rank",
      ...cfg.reranker.llamaArgs,
      "--host", cfg.upstreamHost,
      "--port", String(cfg.reranker.upstreamPort),
      "--no-webui",
    ];
  }
  return [
    "-m", cfg.modelPath,
    "--alias", cfg.modelId,
    "--embeddings",
    "--pooling", "last",
    ...cfg.modes[mode].llamaArgs,
    "--host", cfg.upstreamHost,
    "--port", String(cfg.upstreamPort),
    "--no-webui",
  ];
}

/** Does a `ps` command line describe the child this config owns? */
export function commandLineMatches(
  command: string,
  cfg: EmbedConfig,
  kind: UpstreamKind = "embedding",
): boolean {
  return (
    command.includes("llama-server") &&
    command.includes(modelPathFor(cfg, kind)) &&
    command.includes(String(portFor(cfg, kind)))
  );
}

function truncateIfLarge(path: string): void {
  try {
    if (existsSync(path) && statSync(path).size > LOG_LIMIT_BYTES) rmSync(path, { force: true });
  } catch {
    /* a log that cannot be trimmed is not worth failing a start over */
  }
}

function readLogTail(path: string, lines = 12): string {
  try {
    if (!existsSync(path)) return "";
    return readFileSync(path, "utf8").split("\n").slice(-lines).join("\n").trim();
  } catch {
    return "";
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Bun 1.4 rejects a `FileSink` in `stdout`/`stderr`, so the child is spawned
 * with pipes and this copies each stream to its log file.
 */
async function pipeTo(
  stream: ReadableStream<Uint8Array> | undefined,
  sink: ReturnType<ReturnType<typeof Bun.file>["writer"]>,
): Promise<void> {
  if (!stream) {
    try {
      sink.end();
    } catch {
      /* already closed */
    }
    return;
  }
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) sink.write(value);
    }
  } catch {
    /* the child closed the pipe */
  } finally {
    try {
      sink.end();
    } catch {
      /* already closed */
    }
  }
}

export class Upstream implements UpstreamLike {
  private child: ReturnType<typeof Bun.spawn> | null = null;
  private childPid: number | null = null;
  private startedAt: number | null = null;
  private currentMode: Mode | null = null;
  private state: UpstreamState = "stopped";
  private startPromise: Promise<void> | null = null;
  private outSink: ReturnType<ReturnType<typeof Bun.file>["writer"]> | null = null;
  private errSink: ReturnType<ReturnType<typeof Bun.file>["writer"]> | null = null;

  constructor(
    private readonly cfg: EmbedConfig,
    private readonly kind: UpstreamKind = "embedding",
  ) {}

  status(): UpstreamStatus {
    return {
      state: this.state,
      kind: this.kind,
      ...(this.childPid !== null ? { pid: this.childPid } : {}),
      ...(this.startedAt !== null ? { startedAt: this.startedAt } : {}),
      ...(this.currentMode !== null ? { mode: this.currentMode } : {}),
    };
  }

  isRunning(): boolean {
    return this.state === "running" && this.child !== null;
  }

  async start(mode: Mode): Promise<void> {
    if (this.isRunning() && this.currentMode === mode) return;
    if (this.startPromise) return this.startPromise;
    if (this.isRunning()) await this.stop();
    this.startPromise = this.spawnAndWait(mode).finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  private installHint(): string {
    return this.kind === "reranker"
      ? "run `gbrain-embed model install reranker`"
      : "run `gbrain-embed model install`";
  }

  private label(): string {
    return this.kind === "reranker" ? "reranker" : "embedding server";
  }

  private async spawnAndWait(mode: Mode): Promise<void> {
    const modelPath = modelPathFor(this.cfg, this.kind);
    if (!existsSync(modelPath)) {
      throw new Error(
        `${this.label()} model file is missing: ${modelPath}; ${this.installHint()}`,
      );
    }
    if (!existsSync(this.cfg.llamaServer)) {
      throw new Error(
        `llama-server is missing: ${this.cfg.llamaServer}; install it with \`brew install llama.cpp\``,
      );
    }
    mkdirSync(this.cfg.stateDir, { recursive: true });
    const paths = pathsFor(this.cfg, this.kind);
    truncateIfLarge(paths.outLogPath);
    truncateIfLarge(paths.errLogPath);

    const args = buildLlamaArgs(this.cfg, mode, this.kind);
    const taskpolicy =
      (this.kind === "reranker" ? this.cfg.reranker.taskpolicy : this.cfg.modes[mode].taskpolicy) &&
      existsSync("/usr/sbin/taskpolicy");
    const cmd = taskpolicy
      ? ["/usr/sbin/taskpolicy", "-b", this.cfg.llamaServer, ...args]
      : [this.cfg.llamaServer, ...args];

    this.state = "starting";
    this.currentMode = mode;
    this.outSink = Bun.file(paths.outLogPath).writer();
    this.errSink = Bun.file(paths.errLogPath).writer();
    try {
      this.child = Bun.spawn({
        cmd,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env },
      });
    } catch (error) {
      this.state = "stopped";
      throw error;
    }
    this.childPid = this.child.pid;
    this.startedAt = Date.now();
    this.writeRecord(mode);
    if (this.outSink) void pipeTo(this.child.stdout, this.outSink);
    if (this.errSink) void pipeTo(this.child.stderr, this.errSink);

    this.child.exited
      .then((code) => {
        if (this.state === "running" || this.state === "starting") {
          this.state = "stopped";
        }
        this.child = null;
        this.childPid = null;
        this.startedAt = null;
        this.endSinks();
        this.removeRecord();
        if (code !== 0) {
          console.error(
            `[gbrain-embed] ${this.label()} exited with ${code}: ${readLogTail(paths.errLogPath)}`,
          );
        }
      })
      .catch(() => {
        this.state = "stopped";
      });

    try {
      await this.waitForHealth();
      this.state = "running";
    } catch (error) {
      await this.stop();
      throw error;
    }
  }

  private async waitForHealth(): Promise<void> {
    const deadline = Date.now() + this.cfg.startupTimeoutMs;
    const healthUrl = `${modelBase(this.cfg, this.kind)}/health`;
    const errLog = pathsFor(this.cfg, this.kind).errLogPath;
    while (Date.now() < deadline) {
      if (this.child === null) {
        throw new Error(
          `${this.label()} exited during startup: ${readLogTail(errLog)}`,
        );
      }
      try {
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1_000) });
        if (response.ok) return;
      } catch {
        /* not listening yet */
      }
      await sleep(100);
    }
    throw new Error(
      `${this.label()} did not become healthy within ${this.cfg.startupTimeoutMs} ms: ` +
        readLogTail(errLog),
    );
  }

  async stop(): Promise<void> {
    const child = this.child;
    if (!child) {
      this.state = "stopped";
      return;
    }
    this.state = "stopping";
    try {
      child.kill("SIGTERM");
    } catch {
      /* already gone */
    }
    const exited = await Promise.race([
      child.exited.then(() => true),
      sleep(STOP_GRACE_MS).then(() => false),
    ]);
    if (!exited) {
      try {
        child.kill("SIGKILL");
      } catch {
        /* already gone */
      }
      await Promise.race([child.exited, sleep(1_000)]);
    }
    this.child = null;
    this.childPid = null;
    this.startedAt = null;
    this.state = "stopped";
    this.endSinks();
    this.removeRecord();
  }

  /**
   * Kill a child left behind by a previous supervisor process. Only a process
   * whose command line names our model, our port and `llama-server` is touched.
   */
  recoverOrphan(): void {
    const record = this.readRecord();
    if (!record) return;
    if (
      record.modelPath !== modelPathFor(this.cfg, this.kind) ||
      record.port !== portFor(this.cfg, this.kind)
    ) {
      this.removeRecord();
      return;
    }
    let command: string;
    try {
      const result = Bun.spawnSync({
        cmd: ["ps", "-p", String(record.pid), "-o", "command="],
        stdout: "pipe",
        stderr: "pipe",
      });
      command = result.stdout.toString().trim();
    } catch (error) {
      // `ps` can be unavailable in a sandbox; never kill on an unverifiable identity.
      console.error(`[gbrain-embed] could not verify orphan ${this.label()} pid ${record.pid}: ${String(error)}`);
      return;
    }
    if (!commandLineMatches(command, this.cfg, this.kind)) {
      this.removeRecord();
      return;
    }
    try {
      process.kill(record.pid, "SIGTERM");
    } catch {
      this.removeRecord();
      return;
    }
    const deadline = Date.now() + ORPHAN_GRACE_MS;
    while (Date.now() < deadline) {
      if (!this.pidAlive(record.pid)) break;
      Bun.sleepSync(50);
    }
    if (this.pidAlive(record.pid)) {
      try {
        process.kill(record.pid, "SIGKILL");
      } catch {
        /* gone */
      }
    }
    this.removeRecord();
    console.error(`[gbrain-embed] stopped orphaned ${this.label()} pid ${record.pid}`);
  }

  /** Resident set size of the child, in bytes, or null when stopped. */
  rssBytes(): number | null {
    if (this.childPid === null) return null;
    try {
      const result = Bun.spawnSync({
        cmd: ["ps", "-o", "rss=", "-p", String(this.childPid)],
        stdout: "pipe",
        stderr: "pipe",
      });
      const kb = Number(result.stdout.toString().trim());
      return Number.isFinite(kb) && kb > 0 ? kb * 1024 : null;
    } catch {
      // `/health` must keep answering even where `ps` is unavailable.
      return null;
    }
  }

  private pidAlive(pid: number): boolean {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  }

  private writeRecord(mode: Mode): void {
    if (this.childPid === null || this.startedAt === null) return;
    const record: ChildRecord = {
      pid: this.childPid,
      startedAt: this.startedAt,
      mode,
      kind: this.kind,
      modelPath: modelPathFor(this.cfg, this.kind),
      port: portFor(this.cfg, this.kind),
    };
    try {
      writeFileSync(pathsFor(this.cfg, this.kind).statePath, `${JSON.stringify(record)}\n`, {
        mode: 0o600,
      });
    } catch (error) {
      console.error(
        `[gbrain-embed] could not write ${pathsFor(this.cfg, this.kind).statePath}: ${String(error)}`,
      );
    }
  }

  private readRecord(): ChildRecord | null {
    try {
      const path = pathsFor(this.cfg, this.kind).statePath;
      if (!existsSync(path)) return null;
      const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<ChildRecord>;
      if (
        !Number.isInteger(parsed.pid) ||
        !Number.isInteger(parsed.port) ||
        typeof parsed.modelPath !== "string"
      ) {
        return null;
      }
      // Records written before the reranker existed have no `kind`; they can
      // only be embedding children.
      const kind = parsed.kind ?? "embedding";
      if (kind !== this.kind) return null;
      return parsed as ChildRecord;
    } catch {
      return null;
    }
  }

  private removeRecord(): void {
    try {
      rmSync(pathsFor(this.cfg, this.kind).statePath, { force: true });
    } catch {
      /* best effort */
    }
  }

  private endSinks(): void {
    for (const sink of [this.outSink, this.errSink]) {
      try {
        sink?.end();
      } catch {
        /* already closed */
      }
    }
    this.outSink = null;
    this.errSink = null;
  }
}

/** Directory that holds the children's logs; exposed for docs/tests. */
export function upstreamLogDir(cfg: EmbedConfig): string {
  return join(cfg.stateDir);
}
