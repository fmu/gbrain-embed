# Options shared by the NixOS and nix-darwin modules.
#
# Both modules run the same `gbrain-embed serve` process; only the service
# manager and the defaults for writable state differ. The configuration is
# rendered to one JSON file (`configFile`) which is passed to the CLI with
# `--config` so the declarative options are the single source of truth.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.gbrain-embed;

  # Values that always come from a dedicated option. `null` means "omit it and
  # let the CLI apply its own default".
  baseConfig = {
    listenHost = cfg.listenHost;
    listenPort = cfg.listenPort;
    upstreamHost = cfg.upstreamHost;
    upstreamPort = cfg.upstreamPort;
    llamaServer = cfg.llamaServer;
    gbrainBin = cfg.gbrainBin;
    modelPath = cfg.modelPath;
    modelId = cfg.modelId;
    nativeDimensions = cfg.nativeDimensions;
    dimensions = cfg.dimensions;
    defaultMode = cfg.mode;
    stateDir = cfg.stateDir;
  };

  effectiveConfig = lib.recursiveUpdate (lib.filterAttrs (_: value: value != null) baseConfig) cfg.settings;
in
{
  options.services.gbrain-embed = {
    enable = lib.mkEnableOption "the gbrain-embed on-demand embedding proxy";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ../package.nix { };
      defaultText = lib.literalExpression "pkgs.callPackage ../package.nix { }";
      description = "The gbrain-embed package to run.";
    };

    installPackage = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Whether to add the package to `environment.systemPackages`.";
    };

    mode = lib.mkOption {
      type = lib.types.enum [
        "background"
        "throughput"
      ];
      default = "background";
      description = ''
        llama.cpp resource profile used on the next model start. `background`
        keeps the context, thread count and idle TTL small; `throughput` trades
        memory for large batches.
      '';
    };

    listenHost = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address the OpenAI-compatible proxy listens on.";
    };

    listenPort = lib.mkOption {
      type = lib.types.port;
      default = 8082;
      description = "Port the OpenAI-compatible proxy listens on.";
    };

    upstreamHost = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address the managed `llama-server` listens on.";
    };

    upstreamPort = lib.mkOption {
      type = lib.types.port;
      default = 8083;
      description = "Port the managed `llama-server` listens on.";
    };

    llamaServer = lib.mkOption {
      type = lib.types.str;
      default = "${pkgs.llama-cpp}/bin/llama-server";
      defaultText = lib.literalExpression ''"''${pkgs.llama-cpp}/bin/llama-server"'';
      description = ''
        `llama-server` executable (llama.cpp). Defaults to the copy from
        nixpkgs; override it to use a different llama.cpp build.
      '';
    };

    gbrainBin = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Path to the `gbrain` CLI. Only needed by the commands that shell out to
        gbrain (`setup`, `model`), not by `serve`. When null, the CLI resolves
        `gbrain` from `PATH`.
      '';
    };

    modelPath = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Path of the Qwen3-Embedding GGUF. When null, the CLI default is used
        (`$HOME/.local/share/gbrain-embed/models/…`; NixOS overrides this).
      '';
    };

    modelId = lib.mkOption {
      type = lib.types.str;
      default = "qwen3-embedding-4b";
      description = "Model identifier reported by `/v1/models` and stored in gbrain's config.";
    };

    nativeDimensions = lib.mkOption {
      type = lib.types.int;
      default = 2560;
      description = "Native width of the GGUF. Qwen3-Embedding-4B emits 2560.";
    };

    dimensions = lib.mkOption {
      type = lib.types.int;
      default = 1536;
      description = "Width the proxy truncates and renormalizes to (Matryoshka prefix).";
    };

    stateDir = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Directory for the mode marker, child PID record and llama-server logs.
        When null, the CLI default is used (`$HOME/.local/state/gbrain-embed`;
        NixOS overrides this).
      '';
    };

    settings = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = ''
        Extra keys for `config.json`, deep-merged over the options above. Use
        this for fields that do not have a dedicated option, for example
        `settings.modes.background.ttlSeconds = 600;`.
      '';
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Environment variables for the service process.";
    };

    configFile = lib.mkOption {
      type = lib.types.path;
      default = pkgs.writeText "gbrain-embed-config.json" (builtins.toJSON effectiveConfig);
      defaultText = lib.literalExpression ''
        pkgs.writeText "gbrain-embed-config.json" (builtins.toJSON …)
      '';
      description = ''
        Rendered `config.json`, passed to the service with `--config`. It is
        also what `gbrain-embed … --config` should point at when installing the
        model or running `setup`.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = lib.mkIf cfg.installPackage [ cfg.package ];

    assertions = [
      {
        assertion = cfg.listenPort != cfg.upstreamPort;
        message = "services.gbrain-embed: listenPort and upstreamPort must differ.";
      }
      {
        assertion = cfg.dimensions <= cfg.nativeDimensions;
        message = "services.gbrain-embed: dimensions must not exceed nativeDimensions.";
      }
    ];
  };
}
