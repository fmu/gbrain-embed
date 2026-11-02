# qwen3-embed

An on-demand local **embedding and reranking** service for Qwen3 models,
served by [llama.cpp](https://github.com/ggml-org/llama.cpp) behind one
loopback, OpenAI-compatible endpoint. Nothing here is specific to a single
consumer: any client that speaks the OpenAI embeddings API can use it. One
intended use is [gbrain](https://github.com/garrytan/gbrain) embedding, where
the same endpoint also serves `POST /v1/rerank` for gbrain's
`llama-server-reranker` recipe.

It is a small loopback proxy that speaks the OpenAI embeddings API and also
serves `POST /v1/rerank` in llama.cpp's rerank shape. Two `llama-server`
children sit behind it, one per model:

- the embedding model (Qwen3-Embedding-4B GGUF), and
- the reranker (Qwen3-Reranker-0.6B GGUF).

Each child starts on the first request that needs it, and each has **its own
idle TTL, port, PID record and log file**. Stopping one never touches the
other, so an idle reranker unloads while an embedder stays resident (and vice
versa), and switching the embedding resource profile does not restart the
reranker. Process exit is the unload: the model's memory goes back to the OS.

llama.cpp makes `--embeddings` and `--reranking` mutually exclusive at server
launch, so the two-model design is also the only one that works: two
`llama-server` processes, one endpoint.

## Features

- **One endpoint, two models** — `/v1/embeddings` and `/v1/rerank` on
  `127.0.0.1:8082`; `/v1/models` lists both ids; `/health` reports both
  children. The embedding route is plain OpenAI; no gbrain is required.
- **Optional gbrain integration** — `setup` points gbrain at the endpoint,
  enables the reranker recipe and backfills, but only when you ask for it.
- **Independent lazy start / idle unload** — the embedding server follows the
  active mode's TTL, the reranker has its own `reranker.ttlSeconds`; each is
  loaded only while it is being used.
- **Two embedding resource profiles** — `background` (small context, few
  threads, background QoS, short TTL) and `throughput` (large batches, more
  threads, long TTL), switchable at runtime without changing the vector space.
  The profiles size the embedding server only; the reranker has a fixed
  launch profile.
- **Matryoshka fitting** — Qwen3-Embedding emits 2560 dimensions natively; the
  proxy can truncate to the configured width (e.g. 1536) and L2-renormalize,
  which keeps a pgvector HNSW index usable.
- **Verified model installs** — hardlinks the Ollama blob when present, or
  downloads the GGUF from Hugging Face, and checks its SHA-256. The reranker
  uses the llama.cpp maintainers' Q8_0 conversion, the only published artifact
  that carries the classifier head (see [Reranker model](#reranker-model-and-quantization)).
- **Loopback only** — `127.0.0.1`, no auth, no egress except the one-time
  model download.

## Requirements

- Optional: the [`gbrain`](https://github.com/garrytan/gbrain) CLI on `PATH`
  (`brew install gbrain` from the tap below), used only by `setup`/`model`;
  `serve` never needs it
- [Bun](https://bun.sh/) ≥ 1.4 (the Homebrew tap and the Nix package install it)
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) with
  reranking support (`--reranking`, `POST /v1/rerank`; a current Homebrew
  build or nixpkgs `llama-cpp` has it)
- The model GGUFs (installed by `qwen3-embed model install`):
  - embedding: `batiai/Qwen3-Embedding-4B-GGUF` Q6_K, ~3.3 GB
  - reranker: `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` Q8_0, ~0.64 GB

## Installation

### Homebrew

The tap carries no Nix references: `bun` and `llama.cpp` come from Homebrew.

```bash
brew tap fmu/homebrew https://github.com/fmu/homebrew.git
brew trust fmu/homebrew
brew install qwen3-embed
brew install gbrain          # optional: only for `qwen3-embed setup`
```

Homebrew derives the clone URL only for `homebrew-*` repositories; this
multi-formula tap is called `homebrew`, so pass the URL explicitly once.
Homebrew 5+ also keeps third-party taps untrusted until `brew trust` names
them.

The formula registers a `brew services` agent for `qwen3-embed serve`:

```bash
qwen3-embed model install      # both GGUFs, SHA-256 verified
qwen3-embed setup              # configure gbrain (embeddings + reranker) and backfill
brew services start qwen3-embed
brew services info qwen3-embed
```

`model install embedding` installs only the embedder and `model install
reranker` only the reranker.

The same tap also carries gbrain itself, needed only for the `setup`
integration:

```bash
brew install gbrain
```

It is not the unrelated npm package: the formula installs upstream's
self-contained binary, leaves the in-place `gbrain upgrade` channel working,
and a scheduled workflow re-pins it to each upstream release so `brew upgrade
gbrain` tracks upstream too.

### Migrating from `gbrain-embed`

The project was called `gbrain-embed` up to v0.2.0. From v0.3.0 the package,
binary, env var and paths are all `qwen3-embed`; existing installs need:

```bash
brew uninstall gbrain-embed
mv ~/.local/share/gbrain-embed ~/.local/share/qwen3-embed
mv ~/.local/state/gbrain-embed  ~/.local/state/qwen3-embed
mv ~/.config/gbrain-embed      ~/.config/qwen3-embed
brew install fmu/tap/qwen3-embed
```

`GBRAIN_EMBED_CONFIG` is now `QWEN3_EMBED_CONFIG`, and the `brew services`
launchd label moves from `name.fmu.gbrain-embed` to `name.fmu.qwen3-embed`.

### Nix

```bash
nix build .#qwen3-embed
./result/bin/qwen3-embed --help
```

For a managed service, use the [NixOS or nix-darwin module](#nix-flake)
instead; module consumers get `llama-server` from nixpkgs and a declarative
`services.qwen3-embed` configuration.

### From source

```bash
bun src/cli.ts --help
```

Write a config (or let the defaults apply):

```json
{
  "llamaServer": "/opt/homebrew/bin/llama-server",
  "modelPath": "/Users/you/.local/share/qwen3-embed/models/qwen3-embedding-4b-q6_k.gguf",
  "modelId": "qwen3-embedding-4b",
  "nativeDimensions": 2560,
  "dimensions": 1536,
  "listenPort": 8082,
  "upstreamPort": 8083,
  "defaultMode": "background",
  "reranker": {
    "enabled": true,
    "modelPath": "/Users/you/.local/share/qwen3-embed/models/qwen3-reranker-0.6b-q8_0.gguf",
    "modelId": "qwen3-reranker-0.6b",
    "upstreamPort": 8084,
    "ttlSeconds": 300,
    "taskpolicy": true,
    "llamaArgs": ["-c", "4096", "-b", "2048", "-ub", "2048", "--parallel", "1", "-t", "4", "-fa", "on"]
  }
}
```

The service reads `~/.config/qwen3-embed/config.json` by default;
`QWEN3_EMBED_CONFIG` or `--config <path>` override it. Missing keys take the
defaults above; a partial `reranker` object is merged over the reranker
defaults. `listenPort`, `upstreamPort` and `reranker.upstreamPort` must all
differ.

```bash
qwen3-embed model install          # both models
qwen3-embed model install reranker # just the reranker
qwen3-embed serve                  # the proxy; normally run by launchd/systemd
qwen3-embed status
qwen3-embed bench                  # both children: cold start, warm throughput, RSS
```

## The two models are independent

| | embedding | reranker |
| --- | --- | --- |
| upstream port (default) | `8083` | `8084` |
| launch flags | `--embeddings --pooling last` | `--reranking --pooling rank` |
| idle TTL | `background`/`throughput` profile (120 s / 1800 s) | `reranker.ttlSeconds` (300 s) |
| state record | `<stateDir>/upstream.json` | `<stateDir>/reranker.json` |
| log files | `llama-server.{out,err}.log` | `llama-reranker.{out,err}.log` |
| started by | `POST /v1/embeddings` | `POST /v1/rerank` |
| stopped by | embedding idle TTL, `mode` switch, `/admin/unload` | reranker idle TTL, `/admin/unload` |

`qwen3-embed mode <background|throughput>` writes the mode and stops the
embedding child so the next embedding request starts it with the new flags;
the reranker keeps running. `POST /admin/unload` with
`{"model":"embedding"|"reranker"|"all"}` targets one child explicitly.
Cold-start requests are also safe: a model that takes longer to load than its
idle TTL is not reaped mid-request.

## gbrain integration (optional)

`qwen3-embed setup` is idempotent and entirely optional: it points gbrain at
this endpoint, enables the embedding and reranker providers, and backfills. It
runs while `gbrain serve` is stopped, because gbrain's PGlite brain has a
single writer:

```bash
qwen3-embed setup
```

It performs:

1. `gbrain config set provider_base_urls.llama-server http://127.0.0.1:8082/v1`
2. `gbrain embeddings enable --embedding-model llama-server:qwen3-embedding-4b --embedding-dimensions 1536`
3. `gbrain config set embedding_query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"`
4. `gbrain config set provider_base_urls.llama-server-reranker http://127.0.0.1:8082/v1`
5. `gbrain config set search.reranker.model llama-server-reranker:qwen3-reranker-0.6b`
6. `gbrain config set search.reranker.enabled true`
7. `gbrain embed --stale --catch-up`
8. `gbrain doctor --only embeddings --json`

Steps 4–6 are skipped when `reranker.enabled` is false, in which case `setup`
disables the reranker in gbrain instead. Both providers point at the same
loopback endpoint; gbrain appends `/embeddings` or `/rerank` itself.

The equivalent manual configuration for both models is:

```bash
gbrain config set provider_base_urls.llama-server http://127.0.0.1:8082/v1
gbrain config set provider_base_urls.llama-server-reranker http://127.0.0.1:8082/v1
gbrain config set search.reranker.model llama-server-reranker:qwen3-reranker-0.6b
gbrain config set search.reranker.enabled true
```

In gbrain's default `balanced` and `tokenmax` search modes the reranker is on;
`conservative` turns it off. `gbrain models doctor` probes the configured
reranker (the first cold start can take a few seconds). The
`llama-server-reranker` recipe is part of gbrain 0.60.x and later; an older
`gbrain` will not know that provider id.

If the brain's vector column is not the configured width, `setup` refuses and
prints the `gbrain migrate embeddings` preview; a width change is a re-embed,
not a config edit.

## Modes

| | `background` | `throughput` |
| --- | --- | --- |
| llama.cpp | `-c 2048 -b 512 -ub 512 --parallel 1 -t 2 -fa on` | `-c 8192 -b 2048 -ub 2048 --parallel 2 -t 8 -fa on` |
| scheduling | `taskpolicy -b` (macOS background QoS) | normal QoS |
| embedding idle TTL | 120 s | 1800 s |
| reranker | unaffected | unaffected |

```bash
qwen3-embed mode throughput
qwen3-embed mode background
```

The mode is persisted under `~/.local/state/qwen3-embed/`; switching stops the
embedding child so the next embedding request starts the new profile. The
reranker has its own `reranker.ttlSeconds` and launch flags and is not part of
the mode profiles.

## CLI

| command | purpose |
| --- | --- |
| `serve` | run the proxy (launchd/systemd runs this) |
| `status [--json]` | mode, both child states, last requests, RSS |
| `mode <background\|throughput>` | switch the embedding resource profile |
| `model install [embedding\|reranker\|all] [--refresh] [--source <...>]` | install and verify the GGUFs |
| `setup [--json]` | configure gbrain (embeddings and reranker) and backfill |
| `bench [--model <embedding\|reranker\|all>] [--mode <m>] [--texts N] [--json]` | measure cold start, warm throughput and RSS |
| `restart [--model <embedding\|reranker\|all>]` | restart the launchd agent (falls back to the proxy admin endpoint) |

`--source` accepts `ollama:<tag>`, `hf:<repo>/<file>` or a filesystem path, and
requires a single model target (it is ambiguous for `all`). The `restart`
command targets the launchd label `name.fmu.qwen3-embed`, the label used by
the reference deployment. Change it in `src/cli.ts` if you use a different
one; with the Homebrew tap use `brew services restart qwen3-embed`.

## HTTP surface

- `POST /v1/embeddings` — OpenAI-compatible; `input` string or array,
  optional `dimensions` (defaults to the configured width).
- `POST /v1/rerank` (alias `POST /rerank`) — llama.cpp rerank shape:
  `{"model": "...", "query": "...", "documents": ["..."], "top_n": 25}`.
  The response is passed through from `llama-server`:
  `{"results": [{"index": 0, "relevance_score": 0.99}, ...]}`. The request
  starts the reranker on demand; `reranker.enabled=false` answers 503.
- `GET /v1/models` — static listing of both model ids; does not wake either
  model.
- `GET /health` — mode, both child states, per-model idle TTLs and last
  request times, RSS. The embedding fields (`upstream`, `lastRequestAt`,
  `idleTtlMs`, `modelPath`, `modelId`, `dimensions`, `nativeDimensions`,
  `rssBytes`) keep their original top-level names; the reranker is reported
  under `reranker`.
- `POST /admin/mode` — `{"mode":"background"|"throughput"}`; stops the
  embedding child only.
- `POST /admin/unload` — `{"model":"embedding"|"reranker"|"all"}` (default
  `all`); drops the selected child so the next request restarts it.
- `POST /admin/restart` — compatibility alias for unloading both children.

## Reranker model and quantization

The shipped reranker is the llama.cpp maintainers' Qwen3-Reranker-0.6B Q8_0
conversion (`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`, 639,153,184 bytes,
SHA-256 `22c9979c…23429a48`). It is the only published artifact we could verify
that contains the classifier tensor `cls.output.weight`; without that tensor
llama-server loads the GGUF but returns degenerate (~1e-9) scores.

We measured Q8_0 against locally re-quantized copies produced from that verified
Q8_0 with `llama-quantize` (Q4_K_M, Q5_K_M, Q6_K) on BEIR SciFact, 60 queries
with lexically confusable hard negatives, plus cold-start, warm latency and RSS.
Full report: [`docs/reranker-benchmark.md`](docs/reranker-benchmark.md); the
scripts that produced it are under [`bench/reranker/`](bench/reranker/).

| variant | nDCG@5 | MRR@10 | warm 30 docs | RSS | file |
| --- | --- | --- | --- | --- | --- |
| **Q8_0 (shipped)** | 0.8871 | 0.8727 | 3.63 s | 1170 MiB | 610 MiB |
| Q4_K_M (local requant) | 0.9033 | 0.8996 | 4.12 s | 911 MiB | 378 MiB |
| Q5_K_M (local requant) | 0.8846 | 0.8699 | 4.67 s | 956 MiB | 424 MiB |
| Q6_K (local requant) | 0.8866 | 0.8769 | 3.46 s | 1012 MiB | 472 MiB |

Quality deltas are inside the 60-query sampling noise; on this Apple-silicon
build the quantized variants buy disk and RSS but are not faster, and the
published community quants are unusable (missing classifier head). So Q8_0 is
the default, and quantization is an optional low-memory path: quantize locally
from the verified Q8_0 file

```bash
llama-quantize --allow-requantize \
  ~/.local/share/qwen3-embed/models/qwen3-reranker-0.6b-q8_0.gguf \
  ~/.local/share/qwen3-embed/models/qwen3-reranker-0.6b-q4_k_m.gguf Q4_K_M
```

then point `reranker.modelPath` at the new file and restart the service.
`model install`/`setup` deliberately only accept the pinned Q8_0 digest, so
install/setup first and swap the file when using a custom quantization. Keep
`cls.output.weight` in high precision if you produce your own file
(`llama-quantize --tensor-type cls.output.weight=Q8_0 ...`).

## Development

```bash
bun test
```

The tests use fake upstream HTTP servers and fake children; no model or
`llama-server` is needed. For a real smoke test, point a throwaway config at
the GGUFs, run `qwen3-embed serve`, and check `/health`, `/v1/models`,
`/v1/embeddings` and `/v1/rerank`.

## Nix flake

The flake exports the package plus a NixOS module and a nix-darwin module. Both
modules expose the same `services.qwen3-embed.*` options, render them to
`/etc/qwen3-embed/config.json`, and run `qwen3-embed serve` under the platform
service manager; `llama-server` comes from nixpkgs.

### Package

```nix
inputs.qwen3-embed.url = "github:fmu/qwen3-embed";
# ...
inputs.qwen3-embed.packages.${system}.qwen3-embed
```

or directly:

```bash
nix run github:fmu/qwen3-embed -- --help
```

The package is a `writeShellApplication` that runs `src/cli.ts` with Bun and
puts `curl`/`coreutils` on `PATH` for the model installer. It does not pin a
`llama-server`; the modules wire the nixpkgs one in for you.

### NixOS

```nix
{
  inputs.qwen3-embed.url = "github:fmu/qwen3-embed";

  outputs = { self, nixpkgs, qwen3-embed, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      modules = [
        qwen3-embed.nixosModules.default
        {
          services.qwen3-embed.enable = true;
          # optional:
          # services.qwen3-embed.mode = "throughput";
          # services.qwen3-embed.settings.modes.background.ttlSeconds = 600;
          # services.qwen3-embed.rerankerTtlSeconds = 600;
          # services.qwen3-embed.rerankerEnable = false;
        }
      ];
    };
  };
}
```

The module creates a `qwen3-embed` system user, keeps the models and state
under `/var/lib/qwen3-embed`, writes `/etc/qwen3-embed/config.json`, and
starts `qwen3-embed.service`. Install the models once as the service user,
then configure gbrain and backfill as yourself:

```bash
sudo -u qwen3-embed /run/current-system/sw/bin/qwen3-embed \
  --config /etc/qwen3-embed/config.json model install
qwen3-embed --config /etc/qwen3-embed/config.json setup
```

`setup` reuses the running service, so it does not need write access to the
service user's state directory. To run the proxy as your own login user instead
of the dedicated account, set `user = "you"; createUser = false;` — the
home-based defaults then apply and `qwen3-embed setup` works without the split
above.

### nix-darwin

```nix
{
  inputs.qwen3-embed.url = "github:fmu/qwen3-embed";
  inputs.nix-darwin.url = "github:nix-darwin/nix-darwin";

  outputs = { self, nix-darwin, qwen3-embed, ... }: {
    darwinConfigurations.mac = nix-darwin.lib.darwinSystem {
      modules = [
        qwen3-embed.darwinModules.default
        {
          system.primaryUser = "you"; # required by nix-darwin for user agents
          services.qwen3-embed.enable = true;
        }
      ];
    };
  };
}
```

The module installs a launchd user agent (`org.nixos.qwen3-embed`), writes the
same `/etc/qwen3-embed/config.json`, and keeps the models and state under your
`$HOME` (the CLI defaults). Install the models and configure gbrain once:

```bash
qwen3-embed --config /etc/qwen3-embed/config.json model install
qwen3-embed --config /etc/qwen3-embed/config.json setup
```

### Options

Both modules share these options:

| option | default | meaning |
| --- | --- | --- |
| `enable` | `false` | run the proxy as a system/launchd service |
| `package` | built from this flake | package to run |
| `installPackage` | `true` | add the package to `environment.systemPackages` |
| `mode` | `"background"` | embedding llama.cpp profile (`background` or `throughput`) |
| `listenHost`, `listenPort` | `127.0.0.1`, `8082` | OpenAI-compatible endpoint |
| `upstreamHost`, `upstreamPort` | `127.0.0.1`, `8083` | managed embedding `llama-server` |
| `rerankerEnable` | `true` | serve and configure the reranker |
| `rerankerUpstreamPort` | `8084` | managed reranker `llama-server` |
| `rerankerModelPath` | `null` | reranker GGUF path; NixOS defaults to `dataDir/models/…` |
| `rerankerModelId` | `"qwen3-reranker-0.6b"` | reranker id reported by `/v1/models` |
| `rerankerTtlSeconds` | `300` | independent reranker idle TTL |
| `rerankerTaskpolicy` | `true` | wrap the reranker in `taskpolicy -b` |
| `rerankerLlamaArgs` | `null` | extra reranker llama.cpp flags |
| `llamaServer` | nixpkgs `llama-cpp` | `llama-server` executable |
| `gbrainBin` | `null` | path to `gbrain`, used by `setup`/`model`, not `serve` |
| `modelPath` | `null` | embedding GGUF path; NixOS defaults to `dataDir/models/…` |
| `stateDir` | `null` | mode/PID/log directory; NixOS defaults to `dataDir/state` |
| `modelId`, `dimensions`, `nativeDimensions` | `qwen3-embedding-4b`, `1536`, `2560` | embedding vector identity |
| `settings` | `{}` | extra `config.json` keys, deep-merged (e.g. `modes`) |
| `environment` | `{}` | extra service environment |
| `configFile` | generated | rendered config passed with `--config` |

NixOS-only: `user`, `group`, `createUser`, `dataDir` (defaults
`qwen3-embed`, `qwen3-embed`, `true`, `/var/lib/qwen3-embed`).

`gbrain` itself is not part of the flake; the proxy only shells out to it from
the `setup` and `model` subcommands, never from `serve`.
