# gbrain-embed

An on-demand local embedding service for [gbrain](https://github.com/garrytan/gbrain).

It is a small loopback proxy that speaks the OpenAI embeddings API. On the
first embedding request it starts `llama-server` (llama.cpp) with a
Qwen3-Embedding GGUF, waits for it to become healthy, forwards the batch, and
stops the server again after an idle window. Process exit is the unload: the
model's memory goes back to the OS.

Because gbrain's `llama-server` recipe has no model lifecycle of its own, this
proxy is what makes a local embedder work "transparently": gbrain points at a
stable URL, and the model is resident only while it is being used.

## Features

- **Lazy start / idle unload** — no model process at rest, no manual
  `llama-server` command.
- **Two resource profiles** — `background` (small context, few threads,
  background QoS, short TTL) and `throughput` (large batches, more threads,
  long TTL), switchable at runtime without changing the vector space.
- **Matryoshka fitting** — Qwen3-Embedding emits 2560 dimensions natively; the
  proxy can truncate to the configured width (e.g. 1536) and L2-renormalize,
  which keeps a pgvector HNSW index usable.
- **Verified model install** — hardlinks the Ollama blob when present, or
  downloads the GGUF from Hugging Face, and checks its SHA-256.
- **Loopback only** — `127.0.0.1`, no auth, no egress.

## Requirements

- [Bun](https://bun.sh/) ≥ 1.4 (the Homebrew tap and the Nix package install it)
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp)
  (`brew install llama.cpp`; the Homebrew tap installs it as a dependency)
- A Qwen3-Embedding GGUF. The default is
  `batiai/Qwen3-Embedding-4B-GGUF` Q6_K, which is also the bytes behind the
  Ollama tag `batiai/qwen3-embedding:4b`.

## Installation

### Homebrew

The tap carries no Nix references: `bun` and `llama.cpp` come from Homebrew.

```bash
brew tap fmu/homebrew https://github.com/fmu/homebrew.git
brew install gbrain-embed
```

Homebrew derives the clone URL only for `homebrew-*` repositories; this
multi-formula tap is called `homebrew`, so pass the URL explicitly once.

The formula registers a `brew services` agent for `gbrain-embed serve`:

```bash
gbrain-embed model install      # 3.3 GB GGUF, SHA-256 verified
gbrain-embed setup              # configure gbrain and backfill
brew services start gbrain-embed
brew services info gbrain-embed
```

`gbrain` itself is not a Homebrew formula; install it separately and keep it on
`PATH` (a global npm/bun install lands in `$(brew --prefix)/bin`).

### Nix

```bash
nix build .#gbrain-embed
./result/bin/gbrain-embed --help
```

### From source

```bash
bun src/cli.ts --help
```

Write a config (or let the defaults apply):

```json
{
  "llamaServer": "/opt/homebrew/bin/llama-server",
  "modelPath": "/Users/you/.local/share/gbrain-embed/models/qwen3-embedding-4b-q6_k.gguf",
  "modelId": "qwen3-embedding-4b",
  "nativeDimensions": 2560,
  "dimensions": 1536,
  "listenPort": 8082,
  "upstreamPort": 8083,
  "defaultMode": "background"
}
```

The service reads `~/.config/gbrain-embed/config.json` by default;
`GBRAIN_EMBED_CONFIG` or `--config <path>` override it.

```bash
gbrain-embed model install          # hardlink the Ollama blob or download the HF GGUF
gbrain-embed serve                  # the proxy; normally run by launchd/systemd
gbrain-embed status
gbrain-embed bench                  # cold start, warm texts/s, RSS
```

## Configuring gbrain

`gbrain-embed setup` is idempotent. It runs while `gbrain serve` is stopped,
because gbrain's PGlite brain has a single writer:

```bash
gbrain-embed setup
```

It performs:

1. `gbrain config set provider_base_urls.llama-server http://127.0.0.1:8082/v1`
2. `gbrain embeddings enable --embedding-model llama-server:qwen3-embedding-4b --embedding-dimensions 1536`
3. `gbrain config set embedding_query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"`
4. `gbrain embed --stale --catch-up`
5. `gbrain doctor --only embeddings --json`

If the brain's vector column is not the configured width, `setup` refuses and
prints the `gbrain migrate embeddings` preview; a width change is a re-embed,
not a config edit.

## Modes

| | `background` | `throughput` |
| --- | --- | --- |
| llama.cpp | `-c 2048 -b 512 -ub 512 --parallel 1 -t 2 -fa on` | `-c 8192 -b 2048 -ub 2048 --parallel 2 -t 8 -fa on` |
| scheduling | `taskpolicy -b` (macOS background QoS) | normal QoS |
| idle TTL | 120 s | 1800 s |

```bash
gbrain-embed mode throughput
gbrain-embed mode background
```

The mode is persisted under `~/.local/state/gbrain-embed/`; switching stops the
child so the next request starts the new profile.

## CLI

| command | purpose |
| --- | --- |
| `serve` | run the proxy (launchd/systemd runs this) |
| `status [--json]` | mode, child state, last request, RSS |
| `mode <background\|throughput>` | switch the resource profile |
| `model install [--refresh] [--source <ollama:tag\|hf:repo/file\|path>]` | install and verify the GGUF |
| `setup [--json]` | configure gbrain and backfill |
| `bench [--mode <m>] [--texts N] [--json]` | measure cold start and warm throughput |
| `restart` | restart the launchd agent (falls back to the proxy admin endpoint) |

The `restart` command targets the launchd label `name.fmu.gbrain-embed`, the
label used by the reference deployment. Change it in `src/cli.ts` if you use a
different one; with the Homebrew tap use `brew services restart gbrain-embed`.

## HTTP surface

- `POST /v1/embeddings` — OpenAI-compatible; `input` string or array,
  optional `dimensions` (defaults to the configured width).
- `GET /v1/models` — static model listing; does not wake the model.
- `GET /health` — mode, child state, RSS, last request.
- `POST /admin/mode` — `{"mode":"background"|"throughput"}`.
- `POST /admin/restart` — stop the child; the next request restarts it.

## Development

```bash
bun test
```

The tests use a fake upstream HTTP server; no model or `llama-server` is
needed.

## Consuming the flake

```nix
inputs.gbrain-embed.url = "github:<owner>/gbrain-embed";
# in a module:
inputs.gbrain-embed.packages.${system}.gbrain-embed
```

The package is a `writeShellApplication` that runs `src/cli.ts` with Bun and
puts `curl`/`coreutils` on `PATH` for the model installer.
