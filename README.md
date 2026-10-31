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

- The [`gbrain`](https://github.com/garrytan/gbrain) CLI on `PATH`
  (`brew install gbrain` from the tap below)
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
brew trust fmu/homebrew
brew install gbrain gbrain-embed
```

Homebrew derives the clone URL only for `homebrew-*` repositories; this
multi-formula tap is called `homebrew`, so pass the URL explicitly once.
Homebrew 5+ also keeps third-party taps untrusted until `brew trust` names
them.

The formula registers a `brew services` agent for `gbrain-embed serve`:

```bash
gbrain-embed model install      # 3.3 GB GGUF, SHA-256 verified
gbrain-embed setup              # configure gbrain and backfill
brew services start gbrain-embed
brew services info gbrain-embed
```

The same tap also carries gbrain itself:

```bash
brew install gbrain
```

It is not the unrelated npm package: the formula installs upstream's
self-contained binary, leaves the in-place `gbrain upgrade` channel working,
and a scheduled workflow re-pins it to each upstream release so `brew upgrade
gbrain` tracks upstream too.

### Nix

```bash
nix build .#gbrain-embed
./result/bin/gbrain-embed --help
```

For a managed service, use the [NixOS or nix-darwin module](#nix-flake)
instead; module consumers get `llama-server` from nixpkgs and a declarative
`services.gbrain-embed` configuration.

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

## Nix flake

The flake exports the package plus a NixOS module and a nix-darwin module. Both
modules expose the same `services.gbrain-embed.*` options, render them to
`/etc/gbrain-embed/config.json`, and run `gbrain-embed serve` under the platform
service manager; `llama-server` comes from nixpkgs.

### Package

```nix
inputs.gbrain-embed.url = "github:fmu/gbrain-embed";
# ...
inputs.gbrain-embed.packages.${system}.gbrain-embed
```

or directly:

```bash
nix run github:fmu/gbrain-embed -- --help
```

The package is a `writeShellApplication` that runs `src/cli.ts` with Bun and
puts `curl`/`coreutils` on `PATH` for the model installer. It does not pin a
`llama-server`; the modules wire the nixpkgs one in for you.

### NixOS

```nix
{
  inputs.gbrain-embed.url = "github:fmu/gbrain-embed";

  outputs = { self, nixpkgs, gbrain-embed, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      modules = [
        gbrain-embed.nixosModules.default
        {
          services.gbrain-embed.enable = true;
          # optional:
          # services.gbrain-embed.mode = "throughput";
          # services.gbrain-embed.settings.modes.background.ttlSeconds = 600;
        }
      ];
    };
  };
}
```

The module creates a `gbrain-embed` system user, keeps the model and state under
`/var/lib/gbrain-embed`, writes `/etc/gbrain-embed/config.json`, and starts
`gbrain-embed.service`. Install the 3.3 GB model once as the service user, then
configure gbrain and backfill as yourself:

```bash
sudo -u gbrain-embed /run/current-system/sw/bin/gbrain-embed \
  --config /etc/gbrain-embed/config.json model install
gbrain-embed --config /etc/gbrain-embed/config.json setup
```

`setup` reuses the running service, so it does not need write access to the
service user's state directory. To run the proxy as your own login user instead
of the dedicated account, set `user = "you"; createUser = false;` — the
home-based defaults then apply and `gbrain-embed setup` works without the split
above.

### nix-darwin

```nix
{
  inputs.gbrain-embed.url = "github:fmu/gbrain-embed";
  inputs.nix-darwin.url = "github:nix-darwin/nix-darwin";

  outputs = { self, nix-darwin, gbrain-embed, ... }: {
    darwinConfigurations.mac = nix-darwin.lib.darwinSystem {
      modules = [
        gbrain-embed.darwinModules.default
        {
          system.primaryUser = "you"; # required by nix-darwin for user agents
          services.gbrain-embed.enable = true;
        }
      ];
    };
  };
}
```

The module installs a launchd user agent (`org.nixos.gbrain-embed`), writes the
same `/etc/gbrain-embed/config.json`, and keeps the model and state under your
`$HOME` (the CLI defaults). Install the model and configure gbrain once:

```bash
gbrain-embed --config /etc/gbrain-embed/config.json model install
gbrain-embed --config /etc/gbrain-embed/config.json setup
```

### Options

Both modules share these options:

| option | default | meaning |
| --- | --- | --- |
| `enable` | `false` | run the proxy as a system/launchd service |
| `package` | built from this flake | package to run |
| `installPackage` | `true` | add the package to `environment.systemPackages` |
| `mode` | `"background"` | llama.cpp profile (`background` or `throughput`) |
| `listenHost`, `listenPort` | `127.0.0.1`, `8082` | OpenAI-compatible endpoint |
| `upstreamHost`, `upstreamPort` | `127.0.0.1`, `8083` | managed `llama-server` |
| `llamaServer` | nixpkgs `llama-cpp` | `llama-server` executable |
| `gbrainBin` | `null` | path to `gbrain`, used by `setup`/`model`, not `serve` |
| `modelPath` | `null` | GGUF path; NixOS defaults to `dataDir/models/…` |
| `stateDir` | `null` | mode/PID/log directory; NixOS defaults to `dataDir/state` |
| `modelId`, `dimensions`, `nativeDimensions` | `qwen3-embedding-4b`, `1536`, `2560` | vector identity |
| `settings` | `{}` | extra `config.json` keys, deep-merged (e.g. `modes`) |
| `environment` | `{}` | extra service environment |
| `configFile` | generated | rendered config passed with `--config` |

NixOS-only: `user`, `group`, `createUser`, `dataDir` (defaults
`gbrain-embed`, `gbrain-embed`, `true`, `/var/lib/gbrain-embed`).

`gbrain` itself is not part of the flake; the proxy only shells out to it from
the `setup` and `model` subcommands, never from `serve`.
