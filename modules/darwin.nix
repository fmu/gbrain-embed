{ config, lib, pkgs, ... }:

let
  cfg = config.services.qwen3-embed;
  configPath = "/etc/qwen3-embed/config.json";
  logDir = "${config.system.primaryUserHome}/Library/Logs";
in
{
  imports = [ ./common.nix ];

  config = lib.mkIf cfg.enable {
    environment.etc."qwen3-embed/config.json".source = cfg.configFile;

    # A per-user agent: the proxy must run as the user who owns the model file
    # and the local data under `$HOME`. nix-darwin requires
    # `system.primaryUser` to be set for these agents.
    launchd.user.agents.qwen3-embed = {
      command = "${cfg.package}/bin/qwen3-embed --config ${configPath} serve";
      environment = cfg.environment;
      serviceConfig = {
        KeepAlive = true;
        RunAtLoad = true;
        ProcessType = "Background";
        ThrottleInterval = 10;
        StandardOutPath = "${logDir}/qwen3-embed.log";
        StandardErrorPath = "${logDir}/qwen3-embed.err.log";
      };
    };
  };
}
