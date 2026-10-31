{ config, lib, pkgs, ... }:

let
  cfg = config.services.gbrain-embed;
  configPath = "/etc/gbrain-embed/config.json";
in
{
  imports = [ ./common.nix ];

  options.services.gbrain-embed = {
    user = lib.mkOption {
      type = lib.types.str;
      default = "gbrain-embed";
      description = "User the proxy runs as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "gbrain-embed";
      description = "Group the proxy runs as.";
    };

    createUser = lib.mkOption {
      type = lib.types.bool;
      default = cfg.user == "gbrain-embed";
      defaultText = lib.literalExpression ''true unless `user` is overridden'';
      description = ''
        Whether to create the system user and group. Set this to false when
        `user` already exists (for example when running the proxy as your own
        login user).
      '';
    };

    dataDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/gbrain-embed";
      description = ''
        Writable state directory. It holds the verified GGUF by default and is
        the service's working directory and `HOME`.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Keep the model inside the service user's data directory instead of a
    # `$HOME/.local/share` that may not be meaningful for a system user.
    services.gbrain-embed.modelPath = lib.mkDefault "${cfg.dataDir}/models/qwen3-embedding-4b-q6_k.gguf";
    services.gbrain-embed.stateDir = lib.mkDefault "${cfg.dataDir}/state";

    environment.etc."gbrain-embed/config.json".source = cfg.configFile;

    # Always provide the group: the unit and the data directory reference it
    # even when the user itself is managed outside this module.
    users.groups.${cfg.group} = { };

    users.users.${cfg.user} = lib.mkIf cfg.createUser {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      # The human operator runs `gbrain-embed setup` against this config, and
      # that command reads (never rewrites) the model file.
      homeMode = "0755";
      createHome = true;
      description = "gbrain-embed service user";
    };

    # Make sure the data directory exists with the right ownership even when
    # `createUser = false` and the user is managed elsewhere.
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 ${cfg.user} ${cfg.group} -"
    ];

    systemd.services.gbrain-embed = {
      description = "gbrain-embed on-demand embedding proxy";
      documentation = [ "https://github.com/fmu/gbrain-embed" ];
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        HOME = cfg.dataDir;
      } // cfg.environment;

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/gbrain-embed --config ${configPath} serve";
        Restart = "always";
        RestartSec = 2;
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;

        # The proxy only needs to read its config, the model and llama.cpp
        # from the store and write inside its own data directory.
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "full";
      };
    };
  };
}
